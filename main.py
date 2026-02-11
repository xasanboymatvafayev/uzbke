#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIESTA FOOD DELIVERY BOT - PRODUCTION READY
Python 3.11+ | aiogram 3.x | PostgreSQL | Redis | FastAPI
WEBHOOK MODE - FULLY WORKING VERSION
"""

import asyncio
import logging
import os
import json
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import parse_qs, unquote

# aiogram
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    Message, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo, Update
)
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage

# FastAPI
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    DateTime, ForeignKey, BigInteger, Text, select, func, update, delete
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.pool import NullPool

# Redis
import redis.asyncio as aioredis

# ==================== CONFIGURATION ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4")
ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "6365371142").split(",") if x.strip()]
DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411")
SHOP_CHANNEL_ID = int(os.getenv("SHOP_CHANNEL_ID", "-1003530497437"))
COURIER_CHANNEL_ID = int(os.getenv("COURIER_CHANNEL_ID", "-1003707946746"))
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.vercel.app")
PORT = int(os.getenv("PORT", "8000"))

# Webhook configuration
WEBHOOK_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN", os.getenv("RENDER_EXTERNAL_HOSTNAME", "uzbke-production.up.railway.app"))
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"
WEBHOOK_URL = f"https://{WEBHOOK_DOMAIN}{WEBHOOK_PATH}"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATABASE MODELS ====================
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    tg_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(255))
    full_name = Column(String(255))
    joined_at = Column(DateTime, default=datetime.utcnow)
    ref_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    promo_given = Column(Boolean, default=False)
    
    orders = relationship("Order", back_populates="user")
    referrals = relationship("User", backref="referrer", remote_side=[id])

class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    
    foods = relationship("Food", back_populates="category")

class Food(Base):
    __tablename__ = "foods"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=5.0)
    is_new = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    image_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category", back_populates="foods")

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=False)
    comment = Column(Text)
    total = Column(Float, nullable=False)
    status = Column(String(50), default="NEW")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    location_lat = Column(Float)
    location_lng = Column(Float)
    courier_id = Column(Integer, ForeignKey("couriers.id"), nullable=True)
    admin_message_id = Column(Integer, nullable=True)
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    courier = relationship("Courier", back_populates="orders")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    food_id = Column(Integer, ForeignKey("foods.id"))
    name_snapshot = Column(String(255), nullable=False)
    price_snapshot = Column(Float, nullable=False)
    qty = Column(Integer, nullable=False)
    line_total = Column(Float, nullable=False)
    
    order = relationship("Order", back_populates="items")

class Promo(Base):
    __tablename__ = "promos"
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    discount_percent = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    usage_limit = Column(Integer, default=1)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class Courier(Base):
    __tablename__ = "couriers"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    orders = relationship("Order", back_populates="courier")

# ==================== DATABASE SESSION ====================
engine = None
async_session_maker = None

async def init_db():
    """Initialize database tables"""
    global engine, async_session_maker
    try:
        engine = create_async_engine(DB_URL, echo=False, poolclass=NullPool)
        async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
        return False

async def get_session() -> AsyncSession:
    async with async_session_maker() as session:
        yield session

# ==================== BOT & DISPATCHER ====================
bot = None
storage = None
dp = None
router = Router()

def init_bot():
    """Initialize bot and dispatcher"""
    global bot, storage, dp
    try:
        bot = Bot(token=BOT_TOKEN)
        storage = RedisStorage.from_url(REDIS_URL)
        dp = Dispatcher(storage=storage)
        dp.include_router(router)
        logger.info("✅ Bot initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Bot initialization error: {e}")
        return False

# ==================== SERVICES ====================

async def get_or_create_user(session: AsyncSession, tg_id: int, username: str = None, full_name: str = None, ref_by: int = None) -> User:
    result = await session.execute(select(User).where(User.tg_id == tg_id))
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(
            tg_id=tg_id,
            username=username,
            full_name=full_name,
            ref_by_user_id=ref_by
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        logger.info(f"✅ New user created: {tg_id}")
    
    return user

async def get_referral_stats(session: AsyncSession, user_id: int) -> Dict:
    ref_result = await session.execute(
        select(func.count(User.id)).where(User.ref_by_user_id == user_id)
    )
    ref_count = ref_result.scalar() or 0
    
    referred_users_result = await session.execute(
        select(User.tg_id).where(User.ref_by_user_id == user_id)
    )
    referred_user_tg_ids = [row[0] for row in referred_users_result.all()]
    
    if not referred_user_tg_ids:
        return {"ref_count": 0, "orders_count": 0, "delivered_count": 0}
    
    orders_result = await session.execute(
        select(func.count(Order.id)).where(Order.user_id.in_(
            select(User.id).where(User.tg_id.in_(referred_user_tg_ids))
        ))
    )
    orders_count = orders_result.scalar() or 0
    
    delivered_result = await session.execute(
        select(func.count(Order.id)).where(
            Order.user_id.in_(select(User.id).where(User.tg_id.in_(referred_user_tg_ids))),
            Order.status == "DELIVERED"
        )
    )
    delivered_count = delivered_result.scalar() or 0
    
    return {
        "ref_count": ref_count,
        "orders_count": orders_count,
        "delivered_count": delivered_count
    }

async def create_referral_promo(session: AsyncSession, user: User) -> Promo:
    promo_code = f"REF15_{user.tg_id}"
    promo = Promo(
        code=promo_code,
        discount_percent=15,
        usage_limit=1,
        is_active=True
    )
    session.add(promo)
    user.promo_given = True
    await session.commit()
    await session.refresh(promo)
    return promo

async def get_categories(session: AsyncSession) -> List[Category]:
    result = await session.execute(select(Category).where(Category.is_active == True))
    return result.scalars().all()

async def get_foods_by_category(session: AsyncSession, category_id: int = None) -> List[Food]:
    query = select(Food).where(Food.is_active == True)
    if category_id:
        query = query.where(Food.category_id == category_id)
    result = await session.execute(query.order_by(Food.created_at.desc()))
    return result.scalars().all()

async def validate_promo(session: AsyncSession, code: str) -> Optional[Dict]:
    result = await session.execute(select(Promo).where(Promo.code == code, Promo.is_active == True))
    promo = result.scalar_one_or_none()
    
    if not promo:
        return None
    
    if promo.expires_at and promo.expires_at < datetime.utcnow():
        return None
    
    if promo.used_count >= promo.usage_limit:
        return None
    
    return {
        "code": promo.code,
        "discount_percent": promo.discount_percent
    }

async def create_order(session: AsyncSession, user_id: int, data: Dict) -> Order:
    last_order = await session.execute(select(Order).order_by(Order.id.desc()).limit(1))
    last = last_order.scalar_one_or_none()
    order_number = f"ORD{(last.id + 1 if last else 1):06d}"
    
    order = Order(
        order_number=order_number,
        user_id=user_id,
        customer_name=data["customer_name"],
        phone=data["phone"],
        comment=data.get("comment", ""),
        total=data["total"],
        status="NEW",
        location_lat=data["location"]["lat"],
        location_lng=data["location"]["lng"]
    )
    session.add(order)
    await session.flush()
    
    for item in data["items"]:
        order_item = OrderItem(
            order_id=order.id,
            food_id=item["food_id"],
            name_snapshot=item["name"],
            price_snapshot=item["price"],
            qty=item["qty"],
            line_total=item["price"] * item["qty"]
        )
        session.add(order_item)
    
    await session.commit()
    await session.refresh(order)
    return order

async def get_user_orders(session: AsyncSession, user_id: int, limit: int = 10) -> List[Order]:
    result = await session.execute(
        select(Order).where(Order.user_id == user_id).order_by(Order.created_at.desc()).limit(limit)
    )
    return result.scalars().all()

async def get_active_orders(session: AsyncSession) -> List[Order]:
    result = await session.execute(
        select(Order).where(Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"]))
        .order_by(Order.created_at.desc())
    )
    return result.scalars().all()

async def update_order_status(session: AsyncSession, order_id: int, status: str, courier_id: int = None):
    values = {"status": status, "updated_at": datetime.utcnow()}
    if status == "DELIVERED":
        values["delivered_at"] = datetime.utcnow()
    if courier_id:
        values["courier_id"] = courier_id
    
    await session.execute(update(Order).where(Order.id == order_id).values(**values))
    await session.commit()

async def get_active_couriers(session: AsyncSession) -> List[Courier]:
    result = await session.execute(select(Courier).where(Courier.is_active == True))
    return result.scalars().all()

async def get_stats(session: AsyncSession, period: str = "today") -> Dict:
    now = datetime.utcnow()
    
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    else:
        start = None
    
    query = select(Order)
    if start:
        query = query.where(Order.created_at >= start)
    
    result = await session.execute(query)
    orders = result.scalars().all()
    
    orders_count = len(orders)
    delivered_count = len([o for o in orders if o.status == "DELIVERED"])
    revenue = sum(o.total for o in orders if o.status == "DELIVERED")
    
    active_result = await session.execute(
        select(func.count(Order.id)).where(Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"]))
    )
    active_count = active_result.scalar() or 0
    
    return {
        "period": period,
        "orders_count": orders_count,
        "delivered_count": delivered_count,
        "revenue": revenue,
        "active_orders": active_count
    }

async def notify_user(tg_id: int, text: str):
    try:
        await bot.send_message(tg_id, text)
    except Exception as e:
        logger.error(f"Failed to notify user {tg_id}: {e}")

async def update_admin_post(message_id: int, order: Order, items: List[OrderItem]):
    status_emoji = {
        "NEW": "🆕", "CONFIRMED": "✅", "COOKING": "🍳",
        "COURIER_ASSIGNED": "🚴", "OUT_FOR_DELIVERY": "📦",
        "DELIVERED": "✅", "CANCELED": "❌"
    }
    
    status_text = {
        "NEW": "Принят", "CONFIRMED": "Подтвержден", "COOKING": "Готовится",
        "COURIER_ASSIGNED": "Курьер назначен", "OUT_FOR_DELIVERY": "Передан курьеру",
        "DELIVERED": "Доставлен", "CANCELED": "Отменен"
    }
    
    items_text = "\n".join([f"  • {item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум" for item in items])
    
    text = f"""{status_emoji.get(order.status, '📦')} <b>Заказ №{order.order_number}</b>

👤 <b>Клиент:</b> {order.customer_name}
📞 <b>Телефон:</b> {order.phone}
💰 <b>Сумма:</b> {order.total:,.0f} сум
📦 <b>Статус:</b> {status_text.get(order.status, order.status)}
🕒 <b>Время:</b> {order.created_at.strftime('%d.%m.%Y %H:%M')}

🍽️ <b>Заказ:</b>
{items_text}

📝 <b>Комментарий:</b> {order.comment or 'Нет'}
"""
    
    keyboard = []
    if order.status == "NEW":
        keyboard.append([
            InlineKeyboardButton(text="✅ Подтвердить", callback_data=f"order_confirm:{order.id}"),
            InlineKeyboardButton(text="❌ Отменить", callback_data=f"order_cancel:{order.id}")
        ])
    elif order.status == "CONFIRMED":
        keyboard.append([InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order_cooking:{order.id}")])
    elif order.status == "COOKING":
        keyboard.append([InlineKeyboardButton(text="🚴 Назначить курьера", callback_data=f"order_assign_courier:{order.id}")])
    
    if order.location_lat and order.location_lng:
        keyboard.append([InlineKeyboardButton(text="📍 Локация", url=f"https://maps.google.com/?q={order.location_lat},{order.location_lng}")])
    
    try:
        await bot.edit_message_text(
            text=text,
            chat_id=SHOP_CHANNEL_ID,
            message_id=message_id,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard) if keyboard else None,
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"Failed to update admin post: {e}")

# ==================== KEYBOARDS ====================

def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")]
        ],
        resize_keyboard=True
    )

def get_shop_inline_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))]
        ]
    )

def get_admin_main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🍔 Таомлар", callback_data="admin_foods")],
            [InlineKeyboardButton(text="📂 Категориялар", callback_data="admin_categories")],
            [InlineKeyboardButton(text="🎁 Промокодлар", callback_data="admin_promos")],
            [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
            [InlineKeyboardButton(text="🚴 Курьерлар", callback_data="admin_couriers")],
            [InlineKeyboardButton(text="📦 Актив буюртмалар", callback_data="admin_active_orders")]
        ]
    )

# ==================== CLIENT HANDLERS ====================

@router.message(CommandStart())
async def cmd_start(message: Message):
    try:
        async with async_session_maker() as session:
            args = message.text.split()
            ref_by = None
            if len(args) > 1:
                try:
                    ref_by_tg_id = int(args[1])
                    if ref_by_tg_id != message.from_user.id:
                        ref_user_result = await session.execute(select(User).where(User.tg_id == ref_by_tg_id))
                        ref_user = ref_user_result.scalar_one_or_none()
                        if ref_user:
                            ref_by = ref_user.id
                except:
                    pass
            
            user = await get_or_create_user(
                session,
                tg_id=message.from_user.id,
                username=message.from_user.username,
                full_name=message.from_user.full_name,
                ref_by=ref_by
            )
            
            await message.answer(
                f"Добро пожаловать в FIESTA! {user.full_name}\n\n"
                "Для заказа перейдите по кнопке ➡️ 🛍 Заказать",
                reply_markup=get_main_keyboard()
            )
    except Exception as e:
        logger.error(f"Error in cmd_start: {e}")
        await message.answer("Произошла ошибка. Попробуйте позже.")

@router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    try:
        async with async_session_maker() as session:
            user = await get_or_create_user(session, message.from_user.id)
            orders = await get_user_orders(session, user.id)
            
            if not orders:
                await message.answer(
                    "В данный момент у вас нет активных заказов в нашем магазине.\n\n"
                    "Чтобы открыть магазин, введите команду — /shop"
                )
                return
            
            text = "📦 <b>Ваши заказы:</b>\n\n"
            for order in orders:
                items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order.id))
                items = items_result.scalars().all()
                items_text = "\n".join([f"  • {item.name_snapshot} x{item.qty}" for item in items])
                
                text += f"🆔 <b>Заказ №{order.order_number}</b>\n"
                text += f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                text += f"💰 {order.total:,.0f} сум\n"
                text += f"📦 {order.status}\n"
                text += f"{items_text}\n\n"
            
            await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in my_orders: {e}")

@router.message(Command("shop"))
@router.message(F.text == "🛍 Заказать")
async def shop(message: Message):
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=get_shop_inline_keyboard()
    )

@router.message(F.text == "ℹ️ Информация о нас")
async def info(message: Message):
    text = """🌟 <b>Добро Пожаловать в FIESTA !</b>

📍 <b>Наш адрес:</b> Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи
🏢 <b>Ориентир:</b> Школа №12 Оруджева
📞 <b>Контактный номер:</b> +998 91 420 15 15
🕙 <b>Рабочие часы:</b> 24/7

📷 <b>Мы в Instagram:</b> <a href="https://www.instagram.com/fiesta.khiva">fiesta.khiva</a>
🔗 <b>Найти нас на карте:</b> <a href="https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7">Место расположение</a>
"""
    await message.answer(text, parse_mode="HTML")

@router.message(F.text == "👥 Пригласить друга")
async def referral(message: Message):
    try:
        async with async_session_maker() as session:
            user = await get_or_create_user(session, message.from_user.id)
            stats = await get_referral_stats(session, user.id)
            
            bot_info = await bot.get_me()
            ref_link = f"https://t.me/{bot_info.username}?start={message.from_user.id}"
            
            text = f"""👥 <b>Пригласить друга</b>

За приглашение друга, вы можете получить промо-код от нас

👥 <b>Вы пригласили:</b> {stats['ref_count']} человек
🛒 <b>Оформили заказов:</b> {stats['orders_count']}
💰 <b>Оплатили заказов:</b> {stats['delivered_count']}

👤 <b>Ваша реферальная ссылка:</b>
<code>{ref_link}</code>

Пригласите трех человек и вы получите от нас промо-код со скидкой 15%
"""
            
            if stats['ref_count'] >= 3 and not user.promo_given:
                promo = await create_referral_promo(session, user)
                text += f"\n\n🎉 <b>Поздравляем!</b> Вы получили промокод: <code>{promo.code}</code>"
            
            await message.answer(text, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in referral: {e}")

@router.message(F.web_app_data)
async def webapp_data(message: Message):
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get("type") == "order_create":
            if data["total"] < 50000:
                await message.answer("❌ Минимальная сумма заказа: 50,000 сум")
                return
            
            async with async_session_maker() as session:
                user = await get_or_create_user(session, message.from_user.id)
                order = await create_order(session, user.id, data)
                
                items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order.id))
                items = items_result.scalars().all()
                
                items_text = "\n".join([f"  • {item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум" for item in items])
                await message.answer(
                    f"Ваш заказ принят ✅\n\n"
                    f"🆔 Заказ №{order.order_number}\n"
                    f"💰 Сумма: {order.total:,.0f} сум\n"
                    f"📦 Статус: Принят\n\n"
                    f"🍽️ Заказ:\n{items_text}"
                )
                
                admin_text = f"""🆕 <b>Новый заказ №{order.order_number}</b>

👤 <b>Пользователь:</b> {order.customer_name} (@{message.from_user.username or 'no_username'})
📞 <b>Телефон:</b> {order.phone}
💰 <b>Сумма:</b> {order.total:,.0f} сум
🕒 <b>Время:</b> {order.created_at.strftime('%d.%m.%Y %H:%M')}

🍽️ <b>Заказ:</b>
{items_text}

📝 <b>Комментарий:</b> {order.comment or 'Нет'}
"""
                
                keyboard = [
                    [InlineKeyboardButton(text="✅ Подтвердить", callback_data=f"order_confirm:{order.id}")],
                    [InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order_cooking:{order.id}")],
                    [InlineKeyboardButton(text="🚴 Назначить курьера", callback_data=f"order_assign_courier:{order.id}")]
                ]
                
                if order.location_lat and order.location_lng:
                    keyboard.append([
                        InlineKeyboardButton(text="📍 Локация", url=f"https://maps.google.com/?q={order.location_lat},{order.location_lng}")
                    ])
                
                admin_msg = await bot.send_message(
                    SHOP_CHANNEL_ID,
                    admin_text,
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
                    parse_mode="HTML"
                )
                
                await session.execute(
                    update(Order).where(Order.id == order.id).values(admin_message_id=admin_msg.message_id)
                )
                await session.commit()
                
    except Exception as e:
        logger.error(f"WebApp data error: {e}")
        await message.answer("❌ Произошла ошибка при обработке заказа")

# ==================== ADMIN HANDLERS ====================

@router.message(Command("admin"))
async def admin_panel(message: Message):
    if message.from_user.id not in ADMIN_IDS:
        return
    
    await message.answer(
        "⚙️ <b>Админ панель</b>",
        reply_markup=get_admin_main_keyboard(),
        parse_mode="HTML"
    )

@router.callback_query(F.data == "admin_foods")
async def admin_foods(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    async with async_session_maker() as session:
        foods = await get_foods_by_category(session)
        
        text = "🍔 <b>Таомлар:</b>\n\n"
        for food in foods[:10]:
            text += f"• {food.name} - {food.price:,.0f} сум\n"
        
        keyboard = [
            [InlineKeyboardButton(text="➕ Добавить таом", callback_data="admin_food_add")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]
        ]
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            parse_mode="HTML"
        )

@router.callback_query(F.data == "admin_promos")
async def admin_promos(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    async with async_session_maker() as session:
        result = await session.execute(select(Promo).where(Promo.is_active == True).limit(10))
        promos = result.scalars().all()
        
        text = "🎁 <b>Промокодлар:</b>\n\n"
        for promo in promos:
            text += f"• {promo.code} - {promo.discount_percent}% ({promo.used_count}/{promo.usage_limit})\n"
        
        keyboard = [
            [InlineKeyboardButton(text="➕ Добавить промокод", callback_data="admin_promo_add")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]
        ]
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            parse_mode="HTML"
        )

@router.callback_query(F.data == "admin_stats")
async def admin_stats(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    async with async_session_maker() as session:
        today = await get_stats(session, "today")
        week = await get_stats(session, "week")
        month = await get_stats(session, "month")
        
        text = f"""📊 <b>Статистика</b>

📅 <b>Сегодня:</b>
  • Заказов: {today['orders_count']}
  • Доставлено: {today['delivered_count']}
  • Выручка: {today['revenue']:,.0f} сум
  • Активных: {today['active_orders']}

📅 <b>Неделя:</b>
  • Заказов: {week['orders_count']}
  • Доставлено: {week['delivered_count']}
  • Выручка: {week['revenue']:,.0f} сум

📅 <b>Месяц:</b>
  • Заказов: {month['orders_count']}
  • Доставлено: {month['delivered_count']}
  • Выручка: {month['revenue']:,.0f} сум
"""
        
        keyboard = [[InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]]
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            parse_mode="HTML"
        )

@router.callback_query(F.data == "admin_couriers")
async def admin_couriers(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    async with async_session_maker() as session:
        couriers = await get_active_couriers(session)
        
        text = "🚴 <b>Курьерлар:</b>\n\n"
        for courier in couriers:
            text += f"• {courier.name} (ID: {courier.chat_id})\n"
        
        keyboard = [
            [InlineKeyboardButton(text="➕ Добавить курьера", callback_data="admin_courier_add")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]
        ]
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            parse_mode="HTML"
        )

@router.callback_query(F.data == "admin_active_orders")
async def admin_active_orders(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    async with async_session_maker() as session:
        orders = await get_active_orders(session)
        
        text = "📦 <b>Актив буюртмалар:</b>\n\n"
        for order in orders[:15]:
            text += f"• №{order.order_number} - {order.status} - {order.total:,.0f} сум\n"
        
        keyboard = [[InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]]
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
            parse_mode="HTML"
        )

@router.callback_query(F.data == "admin_back")
async def admin_back(callback: CallbackQuery):
    await callback.message.edit_text(
        "⚙️ <b>Админ панель</b>",
        reply_markup=get_admin_main_keyboard(),
        parse_mode="HTML"
    )

# ==================== ORDER STATUS HANDLERS ====================

@router.callback_query(F.data.startswith("order_confirm:"))
async def order_confirm(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        await update_order_status(session, order_id, "CONFIRMED")
        
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one_or_none()
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        user_result = await session.execute(select(User).where(User.id == order.user_id))
        user = user_result.scalar_one_or_none()
        
        if order.admin_message_id:
            await update_admin_post(order.admin_message_id, order, items)
        
        await notify_user(user.tg_id, f"Ваш заказ №{order.order_number} подтвержден ✅")
        
        await callback.answer("✅ Заказ подтвержден")

@router.callback_query(F.data.startswith("order_cooking:"))
async def order_cooking(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        await update_order_status(session, order_id, "COOKING")
        
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one_or_none()
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        user_result = await session.execute(select(User).where(User.id == order.user_id))
        user = user_result.scalar_one_or_none()
        
        if order.admin_message_id:
            await update_admin_post(order.admin_message_id, order, items)
        
        await notify_user(user.tg_id, f"Ваш заказ №{order.order_number} готовится 🍳")
        
        await callback.answer("🍳 Статус: Готовится")

@router.callback_query(F.data.startswith("order_assign_courier:"))
async def order_assign_courier(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        couriers = await get_active_couriers(session)
        
        if not couriers:
            await callback.answer("❌ Нет активных курьеров", show_alert=True)
            return
        
        keyboard = []
        for courier in couriers:
            keyboard.append([
                InlineKeyboardButton(
                    text=f"🚴 {courier.name}",
                    callback_data=f"assign_courier:{order_id}:{courier.id}"
                )
            ])
        keyboard.append([InlineKeyboardButton(text="◀️ Отмена", callback_data="admin_back")])
        
        await callback.message.edit_text(
            f"Выберите курьера для заказа:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
        )

@router.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier(callback: CallbackQuery):
    if callback.from_user.id not in ADMIN_IDS:
        return
    
    parts = callback.data.split(":")
    order_id = int(parts[1])
    courier_id = int(parts[2])
    
    async with async_session_maker() as session:
        await update_order_status(session, order_id, "COURIER_ASSIGNED", courier_id)
        
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one_or_none()
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        courier_result = await session.execute(select(Courier).where(Courier.id == courier_id))
        courier = courier_result.scalar_one_or_none()
        
        user_result = await session.execute(select(User).where(User.id == order.user_id))
        user = user_result.scalar_one_or_none()
        
        if order.admin_message_id:
            await update_admin_post(order.admin_message_id, order, items)
        
        items_text = "\n".join([f"  • {item.name_snapshot} x{item.qty}" for item in items])
        courier_text = f"""🚴 <b>Новый заказ №{order.order_number}</b>

👤 <b>Клиент:</b> {order.customer_name}
📞 <b>Телефон:</b> {order.phone}
💰 <b>Сумма:</b> {order.total:,.0f} сум

🍽️ <b>Список:</b>
{items_text}

📝 <b>Комментарий:</b> {order.comment or 'Нет'}
"""
        
        courier_keyboard = [
            [InlineKeyboardButton(text="✅ Qabul qildim", callback_data=f"courier_accept:{order.id}")],
            [InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier_delivered:{order.id}")]
        ]
        
        if order.location_lat and order.location_lng:
            courier_keyboard.append([
                InlineKeyboardButton(text="📍 Локация", url=f"https://maps.google.com/?q={order.location_lat},{order.location_lng}")
            ])
        
        await bot.send_message(
            courier.chat_id,
            courier_text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=courier_keyboard),
            parse_mode="HTML"
        )
        
        await notify_user(user.tg_id, f"Ваш заказ №{order.order_number} назначен курьеру 🚴")
        
        await callback.message.edit_text(
            f"✅ Курьер {courier.name} назначен на заказ №{order.order_number}",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[[
                InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")
            ]])
        )

# ==================== COURIER HANDLERS ====================

@router.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        courier_result = await session.execute(select(Courier).where(Courier.chat_id == callback.from_user.id))
        courier = courier_result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("❌ Вы не зарегистрированы как курьер", show_alert=True)
            return
        
        await update_order_status(session, order_id, "OUT_FOR_DELIVERY")
        
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one_or_none()
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        user_result = await session.execute(select(User).where(User.id == order.user_id))
        user = user_result.scalar_one_or_none()
        
        if order.admin_message_id:
            await update_admin_post(order.admin_message_id, order, items)
        
        await notify_user(user.tg_id, f"Ваш заказ №{order.order_number} передан курьеру 🚴")
        
        await callback.answer("✅ Qabul qilindi")

@router.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with async_session_maker() as session:
        courier_result = await session.execute(select(Courier).where(Courier.chat_id == callback.from_user.id))
        courier = courier_result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("❌ Вы не зарегистрированы как курьер", show_alert=True)
            return
        
        await update_order_status(session, order_id, "DELIVERED")
        
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one_or_none()
        
        items_result = await session.execute(select(OrderItem).where(OrderItem.order_id == order_id))
        items = items_result.scalars().all()
        
        user_result = await session.execute(select(User).where(User.id == order.user_id))
        user = user_result.scalar_one_or_none()
        
        if order.admin_message_id:
            await update_admin_post(order.admin_message_id, order, items)
        
        await notify_user(user.tg_id, f"Ваш заказ №{order.order_number} успешно доставлен 🎉\n\nСпасибо!")
        
        await callback.message.edit_reply_markup(reply_markup=None)
        await callback.answer("✅ Yetkazildi!")

# ==================== FASTAPI BACKEND ====================
app = FastAPI(title="FIESTA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "FIESTA Food Delivery API", "version": "1.0.0", "webhook": WEBHOOK_URL}

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/categories")
async def api_categories(session: AsyncSession = Depends(get_session)):
    categories = await get_categories(session)
    return [{"id": c.id, "name": c.name} for c in categories]

@app.get("/api/foods")
async def api_foods(category_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    foods = await get_foods_by_category(session, category_id)
    return [{
        "id": f.id,
        "category_id": f.category_id,
        "name": f.name,
        "description": f.description,
        "price": f.price,
        "rating": f.rating,
        "is_new": f.is_new,
        "image_url": f.image_url
    } for f in foods]

@app.post("/api/promo/validate")
async def api_validate_promo(request: Request, session: AsyncSession = Depends(get_session)):
    data = await request.json()
    code = data.get("code")
    
    if not code:
        raise HTTPException(status_code=400, detail="Code required")
    
    promo = await validate_promo(session, code)
    
    if not promo:
        raise HTTPException(status_code=404, detail="Invalid or expired promo code")
    
    return promo

@app.post(WEBHOOK_PATH)
async def webhook_handler(request: Request):
    """Handle incoming updates from Telegram"""
    try:
        update_data = await request.json()
        telegram_update = Update(**update_data)
        await dp.feed_update(bot, telegram_update)
        return {"ok": True}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"ok": False, "error": str(e)}

# ==================== DEMO DATA SEEDER ====================
async def seed_demo_data():
    """Seed demo categories and foods"""
    try:
        async with async_session_maker() as session:
            result = await session.execute(select(Category))
            if result.scalars().first():
                logger.info("✅ Demo data already exists")
                return
            
            categories_data = [
                "Lavash", "Burger", "Xaggi", "Shaurma", "Hotdog", "Combo", "Sneki", "Sous", "Napitki"
            ]
            
            categories = []
            for name in categories_data:
                cat = Category(name=name, is_active=True)
                session.add(cat)
                categories.append(cat)
            
            await session.flush()
            
            foods_data = {
                "Lavash": [
                    ("Лаваш с курицей", "Сочная курица с овощами", 25000),
                    ("Лаваш с говядиной", "Нежная говядина с соусом", 28000),
                    ("Лаваш мини", "Маленький, но вкусный", 18000)
                ],
                "Burger": [
                    ("Чизбургер", "Классический с сыром", 30000),
                    ("Биг Бургер", "Двойная котлета", 40000),
                    ("Чикен Бургер", "С куриной котлетой", 28000)
                ],
                "Xaggi": [
                    ("Хагги классик", "Оригинальный рецепт", 22000),
                    ("Хагги с сыром", "С расплавленным сыром", 25000),
                    ("Хагги острый", "Для любителей остренького", 24000)
                ],
                "Shaurma": [
                    ("Шаурма куриная", "Сочная курица", 20000),
                    ("Шаурма говяжья", "С говядиной", 23000),
                    ("Шаурма мега", "Большая порция", 30000)
                ],
                "Hotdog": [
                    ("Хот-дог классик", "С сосиской", 12000),
                    ("Хот-дог XXL", "Большой", 18000),
                    ("Хот-дог с сыром", "С сыром чеддер", 15000)
                ],
                "Combo": [
                    ("Комбо №1", "Бургер + фри + напиток", 45000),
                    ("Комбо №2", "Лаваш + фри + напиток", 40000),
                    ("Комбо семейный", "Для всей семьи", 120000)
                ],
                "Sneki": [
                    ("Картофель фри", "Хрустящий", 10000),
                    ("Наггетсы", "5 штук", 15000),
                    ("Луковые кольца", "Порция", 12000)
                ],
                "Sous": [
                    ("Кетчуп", "Классический", 2000),
                    ("Майонез", "Домашний", 2000),
                    ("Острый соус", "Жгучий", 3000)
                ],
                "Napitki": [
                    ("Coca-Cola", "0.5л", 8000),
                    ("Fanta", "0.5л", 8000),
                    ("Сок", "0.33л", 7000)
                ]
            }
            
            for cat in categories:
                if cat.name in foods_data:
                    for name, desc, price in foods_data[cat.name]:
                        food = Food(
                            category_id=cat.id,
                            name=name,
                            description=desc,
                            price=price,
                            rating=4.5 + (hash(name) % 10) / 20,
                            is_active=True,
                            is_new=(hash(name) % 3 == 0)
                        )
                        session.add(food)
            
            await session.commit()
            logger.info("✅ Demo data seeded successfully")
    except Exception as e:
        logger.error(f"❌ Seed data error: {e}")

# ==================== STARTUP & SHUTDOWN ====================
@app.on_event("startup")
async def on_startup():
    """Initialize on startup"""
    try:
        logger.info("🚀 Starting FIESTA Delivery System...")
        
        # Initialize bot
        if not init_bot():
            raise Exception("Bot initialization failed")
        
        # Initialize database
        if not await init_db():
            raise Exception("Database initialization failed")
        
        # Seed demo data
        await seed_demo_data()
        
        # Set webhook
        await bot.delete_webhook(drop_pending_updates=True)
        webhook_info = await bot.set_webhook(
            url=WEBHOOK_URL,
            allowed_updates=["message", "callback_query", "web_app_data"],
            drop_pending_updates=True
        )
        
        bot_info = await bot.get_me()
        
        logger.info(f"✅ Webhook set: {WEBHOOK_URL}")
        logger.info(f"✅ Bot: @{bot_info.username}")
        logger.info(f"✅ API Server: http://0.0.0.0:{PORT}")
        logger.info("✅ System ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise

@app.on_event("shutdown")
async def on_shutdown():
    """Cleanup on shutdown"""
    try:
        if bot:
            await bot.delete_webhook()
            await bot.session.close()
        logger.info("👋 System shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

# ==================== MAIN RUNNER ====================
if __name__ == "__main__":
    logger.info("🎬 Starting FIESTA Delivery Bot...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )
