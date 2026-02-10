#!/usr/bin/env python3
"""
Telegram Food Delivery Bot
Asosiy fayl
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

import redis.asyncio as redis
from sqlalchemy import select, update, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup,
    InlineKeyboardButton, WebAppInfo, ReplyKeyboardMarkup,
    KeyboardButton, ReplyKeyboardRemove, MenuButtonWebApp,
    WebAppData
)
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage

import aiohttp
from pydantic import BaseModel

# Config
BOT_TOKEN = "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4"
ADMIN_IDS = [6365371142]
DB_URL = "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway"
REDIS_URL = "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411"
SHOP_CHANNEL_ID = -1003530497437
COURIER_CHANNEL_ID = -1003707946746
WEBAPP_URL = "https://mainsufooduz.netlify.app"
BACKEND_API_URL = "https://uzbke-production.up.railway.app/api"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Bot va Dispatcher
bot = Bot(token=BOT_TOKEN)
redis_client = redis.from_url(REDIS_URL)
storage = RedisStorage(redis=redis_client)
dp = Dispatcher(storage=storage)

# Routerlar
client_router = Router()
admin_router = Router()
courier_router = Router()

# FSM holatlar
class AdminFoodStates(StatesGroup):
    waiting_for_name = State()
    waiting_for_category = State()
    waiting_for_price = State()
    waiting_for_description = State()

class AdminPromoStates(StatesGroup):
    waiting_for_code = State()
    waiting_for_discount = State()
    waiting_for_limit = State()
    waiting_for_expiry = State()

# Database modellar
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, BigInteger, ForeignKey, DECIMAL
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    tg_id = Column(BigInteger, unique=True, nullable=False)
    username = Column(String(100))
    full_name = Column(String(200), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    ref_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)

class Food(Base):
    __tablename__ = 'foods'
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    price = Column(DECIMAL(10, 2), nullable=False)
    rating = Column(Float, default=4.5)
    is_new = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    image_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category")

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    customer_name = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    comment = Column(Text)
    total = Column(DECIMAL(10, 2), nullable=False)
    status = Column(String(50), default='NEW')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at = Column(DateTime)
    location_lat = Column(Float)
    location_lng = Column(Float)
    courier_id = Column(Integer, ForeignKey('couriers.id'), nullable=True)
    
    user = relationship("User")
    courier = relationship("Courier")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = 'order_items'
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    food_id = Column(Integer, ForeignKey('foods.id'), nullable=False)
    name_snapshot = Column(String(200), nullable=False)
    price_snapshot = Column(DECIMAL(10, 2), nullable=False)
    qty = Column(Integer, nullable=False)
    line_total = Column(DECIMAL(10, 2), nullable=False)
    
    order = relationship("Order", back_populates="items")
    food = relationship("Food")

class Promo(Base):
    __tablename__ = 'promos'
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False)
    discount_percent = Column(Integer, nullable=False)
    expires_at = Column(DateTime)
    usage_limit = Column(Integer, default=100)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class Courier(Base):
    __tablename__ = 'couriers'
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

engine = create_async_engine(DB_URL, echo=False)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Utility funksiyalar
async def get_or_create_user(tg_id: int, username: str, full_name: str, ref_by: int = None):
    async with async_session() as session:
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
            
            # Agar referral bo'lsa
            if ref_by:
                try:
                    ref_user = await session.execute(select(User).where(User.tg_id == ref_by))
                    ref_user_obj = ref_user.scalar_one()
                    # Bu yerda referral statistikani yangilash
                    # Sizga alohida ReferralStat model kerak bo'ladi
                except:
                    pass
        
        return user

def generate_order_number():
    from datetime import datetime
    import random
    date_str = datetime.now().strftime("%Y%m%d")
    random_str = ''.join(random.choices('0123456789', k=6))
    return f"ORD-{date_str}-{random_str}"

# Keyboardlar
def get_client_main_keyboard():
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard

def get_admin_main_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🍔 Taomlar", callback_data="admin:foods")],
            [InlineKeyboardButton(text="📂 Kategoriyalar", callback_data="admin:categories")],
            [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="admin:promos")],
            [InlineKeyboardButton(text="📊 Statistika", callback_data="admin:stats")],
            [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="admin:couriers")],
            [InlineKeyboardButton(text="📦 Aktiv buyurtmalar", callback_data="admin:active_orders")],
            [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="admin:settings")]
        ]
    )
    return keyboard

def get_order_status_keyboard(order_id: int):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"status:confirmed:{order_id}"),
                InlineKeyboardButton(text="🍳 Готовится", callback_data=f"status:cooking:{order_id}")
            ],
            [
                InlineKeyboardButton(text="🚴 Курьер", callback_data=f"status:courier:{order_id}")
            ]
        ]
    )
    return keyboard

def get_courier_choice_keyboard(order_id: int, couriers):
    buttons = []
    for courier in couriers:
        buttons.append([InlineKeyboardButton(
            text=f"🚴 {courier.name}",
            callback_data=f"assign_courier:{order_id}:{courier.id}"
        )])
    
    buttons.append([InlineKeyboardButton(text="⬅️ Ortga", callback_data=f"back_to_order:{order_id}")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_courier_order_keyboard(order_id: int):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Qabul qildim", callback_data=f"courier_accept:{order_id}"),
                InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier_delivered:{order_id}")
            ]
        ]
    )
    return keyboard

# Handlerlar

# ========================
# CLIENT HANDLERS
# ========================

@client_router.message(CommandStart())
async def cmd_start(message: Message):
    args = message.text.split()
    ref_by = None
    
    if len(args) > 1:
        try:
            ref_by = int(args[1])
        except:
            pass
    
    user = await get_or_create_user(
        tg_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
        ref_by=ref_by
    )
    
    welcome_text = (
        f"Добро Пожаловать в FIESTA! {message.from_user.full_name}\n\n"
        f"Для заказа перейдите по кнопке ➡️\n"
        f"🛍 Заказать"
    )
    
    await message.answer(
        welcome_text,
        reply_markup=get_client_main_keyboard()
    )
    
    # Menyu tugmasini o'rnatish
    await bot.set_chat_menu_button(
        chat_id=message.chat.id,
        menu_button=MenuButtonWebApp(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))
    )

@client_router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    async with async_session() as session:
        result = await session.execute(
            select(Order)
            .where(Order.user_id == message.from_user.id)
            .order_by(Order.created_at.desc())
            .limit(10)
        )
        orders = result.scalars().all()
        
        if not orders:
            await message.answer(
                "В данный момент у вас нет активных заказов в нашем магазине.\n"
                "Чтобы открыть магазин, введите команду — /shop"
            )
        else:
            response = "📦 Ваши заказы:\n\n"
            for order in orders:
                status_emoji = {
                    'NEW': '🆕',
                    'CONFIRMED': '✅',
                    'COOKING': '🍳',
                    'COURIER_ASSIGNED': '🚴',
                    'OUT_FOR_DELIVERY': '📦',
                    'DELIVERED': '🎉',
                    'CANCELED': '❌'
                }.get(order.status, '📝')
                
                response += (
                    f"{status_emoji} Заказ №{order.order_number}\n"
                    f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                    f"💰 {order.total} сум\n"
                    f"📊 Статус: {order.status}\n"
                    f"━━━━━━━━━━━━━━\n"
                )
            
            await message.answer(response)

@client_router.message(F.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    about_text = (
        "🌟 Добро Пожаловать в FIESTA !\n\n"
        "📍 Наш адрес: Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢 Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙 Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)\n\n"
        "Мы всегда рады вам! ❤️"
    )
    await message.answer(about_text)

@client_router.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    async with async_session() as session:
        # Referral statistikani hisoblash
        ref_count = 0  # Bu yerda real hisoblash qo'shing
        
        # Buyurtmalar soni
        orders_result = await session.execute(
            select(func.count(Order.id))
            .where(Order.user_id == message.from_user.id)
        )
        orders_count = orders_result.scalar() or 0
        
        # Yetkazilgan buyurtmalar
        delivered_result = await session.execute(
            select(func.count(Order.id))
            .where(Order.user_id == message.from_user.id, Order.status == 'DELIVERED')
        )
        delivered_count = delivered_result.scalar() or 0
        
        referral_link = f"https://t.me/{(await bot.me()).username}?start={message.from_user.id}"
        
        referral_text = (
            "За приглашение друга, вы можете получить промо-код от нас\n\n"
            f"👥 Вы пригласили {ref_count} человек\n"
            f"🛒 Оформили заказов: {orders_count}\n"
            f"💰 Оплатили заказов: {delivered_count}\n\n"
            f"👤 Ваша реферальная ссылка:\n{referral_link}\n\n"
            "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
        )
        
        await message.answer(referral_text)
        
        # Agar 3 tadan ko'p referral bo'lsa
        if ref_count >= 3:
            # Promokod yaratish va tekshirish
            promo_result = await session.execute(
                select(Promo).where(Promo.code.like(f"REF-{message.from_user.id}%"))
            )
            existing_promo = promo_result.scalar_one_or_none()
            
            if not existing_promo:
                # Yangi promo yaratish
                new_promo = Promo(
                    code=f"REF-{message.from_user.id}-{datetime.now().strftime('%m%d')}",
                    discount_percent=15,
                    expires_at=datetime.now() + timedelta(days=30),
                    usage_limit=1
                )
                session.add(new_promo)
                await session.commit()
                
                await message.answer(
                    f"🎉 Поздравляем! Вы получили промо-код: {new_promo.code}\n"
                    f"Скидка: {new_promo.discount_percent}%\n"
                    f"Действует до: {new_promo.expires_at.strftime('%d.%m.%Y')}"
                )

@client_router.message(Command("shop"))
async def cmd_shop(message: Message):
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="🛍 Заказать",
                web_app=WebAppInfo(url=WEBAPP_URL)
            )
        ]]
    )
    
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=keyboard
    )

# ========================
# WEB APP DATA HANDLER
# ========================

@client_router.message(F.web_app_data)
async def handle_web_app_data(message: WebAppData):
    try:
        data = json.loads(message.web_app_data.data)
        logger.info(f"WebApp data received: {data}")
        
        if data.get('type') == 'order_create':
            await process_order_create(message.from_user, data)
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        await message.answer("Ошибка обработки заказа. Попробуйте еще раз.")
    except Exception as e:
        logger.error(f"Error processing web app data: {e}")
        await message.answer("Произошла ошибка. Пожалуйста, попробуйте позже.")

async def process_order_create(user, data: Dict):
    async with async_session() as session:
        # User ni olish
        db_user = await get_or_create_user(
            tg_id=user.id,
            username=user.username,
            full_name=user.full_name
        )
        
        # Total tekshirish
        total = Decimal(str(data['total']))
        if total < 50000:
            await bot.send_message(
                chat_id=user.id,
                text="❌ Минимальная сумма заказа 50,000 сум"
            )
            return
        
        # Promo code tekshirish
        promo_code = data.get('promo_code')
        final_total = total
        
        if promo_code:
            promo_result = await session.execute(
                select(Promo).where(
                    Promo.code == promo_code,
                    Promo.is_active == True,
                    Promo.used_count < Promo.usage_limit,
                    Promo.expires_at > datetime.utcnow()
                )
            )
            promo = promo_result.scalar_one_or_none()
            
            if promo:
                discount = total * Decimal(promo.discount_percent) / 100
                final_total = total - discount
                promo.used_count += 1
            else:
                await bot.send_message(
                    chat_id=user.id,
                    text="❌ Неверный или просроченный промо-код"
                )
                return
        
        # Order yaratish
        order_number = generate_order_number()
        order = Order(
            order_number=order_number,
            user_id=db_user.id,
            customer_name=data['customer_name'],
            phone=data['phone'],
            comment=data.get('comment', ''),
            total=final_total,
            status='NEW',
            location_lat=data['location']['lat'],
            location_lng=data['location']['lng']
        )
        session.add(order)
        await session.flush()
        
        # Order items yaratish
        for item in data['items']:
            order_item = OrderItem(
                order_id=order.id,
                food_id=item['food_id'],
                name_snapshot=item['name'],
                price_snapshot=Decimal(str(item['price'])),
                qty=item['qty'],
                line_total=Decimal(str(item['qty'])) * Decimal(str(item['price']))
            )
            session.add(order_item)
        
        await session.commit()
        await session.refresh(order)
        
        # User ga xabar
        await bot.send_message(
            chat_id=user.id,
            text=(
                "✅ Ваш заказ принят\n\n"
                f"🆔 Заказ №{order.order_number}\n"
                f"💰 Сумма: {order.total} сум\n"
                f"📦 Статус: Принят\n\n"
                "Мы свяжемся с вами для подтверждения заказа."
            )
        )
        
        # Admin kanalga yuborish
        await send_order_to_admin_channel(order)

async def send_order_to_admin_channel(order: Order):
    # Order items olish
    async with async_session() as session:
        result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = ""
        for item in items:
            items_text += f"{item.name_snapshot} x{item.qty} = {item.line_total} сум\n"
    
    order_text = (
        f"🆕 Новый заказ №{order.order_number}\n\n"
        f"👤 Пользователь: {order.customer_name}\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total} сум\n"
        f"🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
        f"📍 Локация: {order.location_lat}, {order.location_lng}\n"
        f"🗺️ Карта: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n\n"
        f"🍽️ Заказ:\n{items_text}"
    )
    
    message = await bot.send_message(
        chat_id=SHOP_CHANNEL_ID,
        text=order_text,
        reply_markup=get_order_status_keyboard(order.id)
    )
    
    # Message ID ni saqlash (keyinchalik edit qilish uchun)
    async with redis_client as r:
        await r.set(f"order_message:{order.id}", message.message_id)

# ========================
# ADMIN HANDLERS
# ========================

@admin_router.message(Command("admin"))
async def admin_panel(message: Message):
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("Доступ запрещен.")
        return
    
    await message.answer(
        "Админ панель:",
        reply_markup=get_admin_main_keyboard()
    )

@admin_router.callback_query(F.data.startswith("admin:"))
async def admin_menu_handler(callback: CallbackQuery):
    action = callback.data.split(":")[1]
    
    if action == "foods":
        await callback.message.edit_text(
            "🍔 Управление блюдами:",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="➕ Добавить блюдо", callback_data="food:add")],
                    [InlineKeyboardButton(text="📝 Список блюд", callback_data="food:list")],
                    [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
                ]
            )
        )
    
    elif action == "active_orders":
        await show_active_orders(callback)
    
    elif action == "back":
        await callback.message.edit_text(
            "Админ панель:",
            reply_markup=get_admin_main_keyboard()
        )
    
    await callback.answer()

async def show_active_orders(callback: CallbackQuery):
    async with async_session() as session:
        result = await session.execute(
            select(Order).where(
                Order.status.in_(['NEW', 'CONFIRMED', 'COOKING', 'COURIER_ASSIGNED', 'OUT_FOR_DELIVERY'])
            ).order_by(Order.created_at.desc()).limit(20)
        )
        orders = result.scalars().all()
        
        if not orders:
            await callback.message.edit_text(
                "Нет активных заказов",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
                    ]
                )
            )
            return
        
        text = "📦 Активные заказы:\n\n"
        keyboard = []
        
        for order in orders:
            status_emoji = {
                'NEW': '🆕',
                'CONFIRMED': '✅',
                'COOKING': '🍳',
                'COURIER_ASSIGNED': '🚴',
                'OUT_FOR_DELIVERY': '📦'
            }.get(order.status, '📝')
            
            text += f"{status_emoji} №{order.order_number} - {order.total} сум\n"
            keyboard.append([
                InlineKeyboardButton(
                    text=f"Открыть №{order.order_number}",
                    callback_data=f"order_detail:{order.id}"
                )
            ])
        
        keyboard.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")])
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard)
        )

@admin_router.callback_query(F.data.startswith("status:"))
async def handle_order_status(callback: CallbackQuery):
    action, order_id = callback.data.split(":")[1], int(callback.data.split(":")[2])
    
    async with async_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one()
        
        if action == "confirmed":
            order.status = "CONFIRMED"
        elif action == "cooking":
            order.status = "COOKING"
        elif action == "courier":
            # Courier tanlash menyusi
            couriers_result = await session.execute(
                select(Courier).where(Courier.is_active == True)
            )
            couriers = couriers_result.scalars().all()
            
            if couriers:
                await callback.message.edit_text(
                    f"Выберите курьера для заказа №{order.order_number}:",
                    reply_markup=get_courier_choice_keyboard(order.id, couriers)
                )
                await callback.answer()
                return
            else:
                await callback.answer("Нет активных курьеров", show_alert=True)
                return
        
        order.updated_at = datetime.utcnow()
        await session.commit()
        
        # Xabarni yangilash
        await update_order_message(order)
        
        await callback.answer(f"Статус изменен на {order.status}")
        await callback.message.delete()

@admin_router.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier_handler(callback: CallbackQuery):
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    async with async_session() as session:
        # Order ni olish
        order_result = await session.execute(select(Order).where(Order.id == order_id))
        order = order_result.scalar_one()
        
        # Courier ni olish
        courier_result = await session.execute(select(Courier).where(Courier.id == courier_id))
        courier = courier_result.scalar_one()
        
        # Yangilash
        order.status = "COURIER_ASSIGNED"
        order.courier_id = courier_id
        order.updated_at = datetime.utcnow()
        await session.commit()
        
        # Admin xabarni yangilash
        await update_order_message(order)
        
        # Kuryerga yuborish
        await send_order_to_courier(order, courier)
        
        # Userga xabar
        await bot.send_message(
            chat_id=order.user.tg_id,
            text=f"✅ Ваш заказ №{order.order_number} принят в работу!"
        )
        
        await callback.answer(f"Курьер {courier.name} назначен")
        await callback.message.delete()

async def update_order_message(order: Order):
    # Eski xabarni olish
    async with redis_client as r:
        message_id = await r.get(f"order_message:{order.id}")
    
    if message_id:
        # Xabarni yangilash
        async with async_session() as session:
            result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = result.scalars().all()
            
            items_text = ""
            for item in items:
                items_text += f"{item.name_snapshot} x{item.qty} = {item.line_total} сум\n"
        
        status_text = {
            'NEW': '🆕 Принят',
            'CONFIRMED': '✅ Подтвержден',
            'COOKING': '🍳 Готовится',
            'COURIER_ASSIGNED': '🚴 Курьер назначен',
            'OUT_FOR_DELIVERY': '📦 Передан курьеру',
            'DELIVERED': '🎉 Доставлен',
            'CANCELED': '❌ Отменен'
        }.get(order.status, order.status)
        
        order_text = (
            f"{'✅ ' if order.status == 'DELIVERED' else ''}Заказ №{order.order_number}\n\n"
            f"👤 Пользователь: {order.customer_name}\n"
            f"📞 Телефон: {order.phone}\n"
            f"💰 Сумма: {order.total} сум\n"
            f"📊 Статус: {status_text}\n"
            f"🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}\n\n"
            f"🍽️ Заказ:\n{items_text}"
        )
        
        if order.status == 'DELIVERED':
            keyboard = None
        else:
            keyboard = get_order_status_keyboard(order.id)
        
        try:
            await bot.edit_message_text(
                chat_id=SHOP_CHANNEL_ID,
                message_id=int(message_id),
                text=order_text,
                reply_markup=keyboard
            )
        except:
            pass  # Xabarni yangilab bo'lmasa

async def send_order_to_courier(order: Order, courier: Courier):
    async with async_session() as session:
        result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = ""
        for item in items:
            items_text += f"• {item.name_snapshot} x{item.qty}\n"
    
    courier_text = (
        f"🚴 Новый заказ №{order.order_number}\n\n"
        f"👤 Клиент: {order.customer_name}\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total} сум\n"
        f"📍 Локация: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n"
        f"🗺️ Навигация: https://yandex.ru/maps/?pt={order.location_lng},{order.location_lat}&z=16\n\n"
        f"🍽️ Список:\n{items_text}\n"
        f"💬 Комментарий: {order.comment if order.comment else 'нет'}"
    )
    
    await bot.send_message(
        chat_id=courier.chat_id,
        text=courier_text,
        reply_markup=get_courier_order_keyboard(order.id)
    )

# ========================
# COURIER HANDLERS
# ========================

@courier_router.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept_order(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with async_session() as session:
        # Order ni olish
        order_result = await session.execute(
            select(Order)
            .where(Order.id == order_id)
        )
        order = order_result.scalar_one()
        
        # Statusni yangilash
        order.status = "OUT_FOR_DELIVERY"
        order.updated_at = datetime.utcnow()
        await session.commit()
        
        # Admin xabarni yangilash
        await update_order_message(order)
        
        # Userga xabar
        await bot.send_message(
            chat_id=order.user.tg_id,
            text=f"✅ Ваш заказ №{order.order_number} передан курьеру 🚴"
        )
        
        await callback.answer("Заказ принят")
        await callback.message.edit_text(
            f"✅ Вы приняли заказ №{order.order_number}\n"
            f"Статус: В пути",
            reply_markup=None
        )

@courier_router.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered_order(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with async_session() as session:
        # Order ni olish
        order_result = await session.execute(
            select(Order)
            .where(Order.id == order_id)
        )
        order = order_result.scalar_one()
        
        # Statusni yangilash
        order.status = "DELIVERED"
        order.delivered_at = datetime.utcnow()
        order.updated_at = datetime.utcnow()
        await session.commit()
        
        # Admin xabarni yangilash
        await update_order_message(order)
        
        # Userga xabar
        await bot.send_message(
            chat_id=order.user.tg_id,
            text=f"🎉 Ваш заказ №{order.order_number} успешно доставлен! Спасибо за заказ!"
        )
        
        await callback.answer("Заказ доставлен")
        await callback.message.edit_text(
            f"✅ Заказ №{order.order_number} доставлен\n"
            f"Время: {order.delivered_at.strftime('%H:%M')}",
            reply_markup=None
        )

# ========================
# MAIN FUNCTION
# ========================

async def main():
    # Routerlarni qo'shish
    dp.include_router(client_router)
    dp.include_router(admin_router)
    dp.include_router(courier_router)
    
    # Database initialization
    await init_db()
    
    logger.info("Bot started...")
    
    # Start polling
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
