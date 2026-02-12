"""
FIESTA Food Delivery Bot
Complete Production-Ready System
Python 3.11+ | aiogram 3.x | PostgreSQL | Redis | FastAPI
"""

import os
import json
import asyncio
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import unquote

# Core imports
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, WebAppInfo, URLInputFile
)

# FastAPI
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Database
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, BigInteger, Text, select, func, update as sql_update
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from redis.asyncio import Redis

# Configuration
from pydantic import BaseModel

# ====================== CONFIGURATION ======================

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN", "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4")
    ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "6365371142").split(",")]
    DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway")
    REDIS_URL = os.getenv("REDIS_URL", "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411")
    SHOP_CHANNEL_ID = int(os.getenv("SHOP_CHANNEL_ID", "-1003530497437"))
    COURIER_CHANNEL_ID = int(os.getenv("COURIER_CHANNEL_ID", "-1003707946746"))
    WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.vercel.app")
    BACKEND_URL = os.getenv("BACKEND_URL", "https://uzbke-production.up.railway.app")
    BOT_USERNAME = None  # Will be set on startup

config = Config()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====================== DATABASE MODELS ======================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    tg_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    full_name = Column(String(255), nullable=False)
    joined_at = Column(DateTime, default=datetime.utcnow)
    ref_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    promo_given = Column(Boolean, default=False)
    
    orders = relationship("Order", back_populates="user")

class Category(Base):
    __tablename__ = "categories"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, default=True)
    
    foods = relationship("Food", back_populates="category")

class Food(Base):
    __tablename__ = "foods"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=5.0)
    is_new = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    image_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category", back_populates="foods")

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=False)
    comment = Column(Text, nullable=True)
    total = Column(Float, nullable=False)
    status = Column(String(50), default="NEW")  # NEW, CONFIRMED, COOKING, COURIER_ASSIGNED, OUT_FOR_DELIVERY, DELIVERED, CANCELED
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at = Column(DateTime, nullable=True)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    courier_id = Column(Integer, ForeignKey("couriers.id"), nullable=True)
    admin_message_id = Column(Integer, nullable=True)
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")
    courier = relationship("Courier", back_populates="orders")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    food_id = Column(Integer, nullable=False)
    name_snapshot = Column(String(255), nullable=False)
    price_snapshot = Column(Float, nullable=False)
    qty = Column(Integer, nullable=False)
    line_total = Column(Float, nullable=False)
    
    order = relationship("Order", back_populates="items")

class Promo(Base):
    __tablename__ = "promos"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), unique=True, nullable=False)
    discount_percent = Column(Integer, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    usage_limit = Column(Integer, nullable=True)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

class Courier(Base):
    __tablename__ = "couriers"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(BigInteger, unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    orders = relationship("Order", back_populates="courier")

# ====================== DATABASE CONNECTION ======================

engine = create_async_engine(config.DB_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")
    
    # Create default categories and foods
    await create_default_data()

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

async def create_default_data():
    """Create default categories and foods"""
    async with AsyncSessionLocal() as session:
        # Check if categories exist
        result = await session.execute(select(Category))
        if result.scalars().first():
            return
        
        # Create categories
        categories_data = [
            "Lavash", "Burger", "Xaggi", "Shaurma", "Hotdog", "Combo", "Sneki", "Sous", "Napitki"
        ]
        
        categories = {}
        for cat_name in categories_data:
            cat = Category(name=cat_name, is_active=True)
            session.add(cat)
            await session.flush()
            categories[cat_name] = cat.id
        
        # Create demo foods
        foods_data = [
            # Lavash
            {"name": "Классик Лаваш", "cat": "Lavash", "price": 25000, "desc": "Курица, помидоры, капуста, соус"},
            {"name": "Сырный Лаваш", "cat": "Lavash", "price": 28000, "desc": "С сыром и курицей"},
            {"name": "Острый Лаваш", "cat": "Lavash", "price": 27000, "desc": "С острым соусом"},
            
            # Burger
            {"name": "Чизбургер", "cat": "Burger", "price": 30000, "desc": "Сочная говядина с сыром"},
            {"name": "Двойной Бургер", "cat": "Burger", "price": 45000, "desc": "Две котлеты, двойной сыр"},
            {"name": "Куриный Бургер", "cat": "Burger", "price": 28000, "desc": "Куриное филе"},
            
            # Xaggi
            {"name": "Хагги Классик", "cat": "Xaggi", "price": 32000, "desc": "Традиционный рецепт"},
            {"name": "Хагги Делюкс", "cat": "Xaggi", "price": 38000, "desc": "С дополнительными ингредиентами"},
            {"name": "Хагги Мега", "cat": "Xaggi", "price": 42000, "desc": "Большая порция"},
            
            # Shaurma
            {"name": "Шаурма по-домашнему", "cat": "Shaurma", "price": 22000, "desc": "Курица, овощи, соус"},
            {"name": "Мега Шаурма", "cat": "Shaurma", "price": 35000, "desc": "Двойная порция"},
            {"name": "Сырная Шаурма", "cat": "Shaurma", "price": 26000, "desc": "С расплавленным сыром"},
            
            # Hotdog
            {"name": "Хот-дог классический", "cat": "Hotdog", "price": 15000, "desc": "Сосиска, булка, соус"},
            {"name": "Хот-дог люкс", "cat": "Hotdog", "price": 20000, "desc": "С дополнительными топпингами"},
            {"name": "Хот-дог XXL", "cat": "Hotdog", "price": 25000, "desc": "Большая порция"},
            
            # Combo
            {"name": "Комбо №1", "cat": "Combo", "price": 55000, "desc": "Бургер + Фри + Напиток"},
            {"name": "Комбо №2", "cat": "Combo", "price": 60000, "desc": "Лаваш + Фри + Напиток"},
            {"name": "Семейный Комбо", "cat": "Combo", "price": 120000, "desc": "Для всей семьи"},
            
            # Sneki
            {"name": "Картофель Фри", "cat": "Sneki", "price": 12000, "desc": "Хрустящий картофель"},
            {"name": "Наггетсы", "cat": "Sneki", "price": 18000, "desc": "Куриные наггетсы (6 шт)"},
            {"name": "Луковые кольца", "cat": "Sneki", "price": 15000, "desc": "Хрустящие кольца"},
            
            # Sous
            {"name": "Кетчуп", "cat": "Sous", "price": 3000, "desc": "Томатный кетчуп"},
            {"name": "Майонез", "cat": "Sous", "price": 3000, "desc": "Классический майонез"},
            {"name": "Чесночный соус", "cat": "Sous", "price": 4000, "desc": "Острый чесночный"},
            
            # Napitki
            {"name": "Coca-Cola 0.5л", "cat": "Napitki", "price": 8000, "desc": "Освежающий напиток"},
            {"name": "Fanta 0.5л", "cat": "Napitki", "price": 8000, "desc": "Апельсиновый напиток"},
            {"name": "Вода 0.5л", "cat": "Napitki", "price": 5000, "desc": "Питьевая вода"},
        ]
        
        for food_data in foods_data:
            food = Food(
                category_id=categories[food_data["cat"]],
                name=food_data["name"],
                description=food_data["desc"],
                price=food_data["price"],
                rating=4.5 + (hash(food_data["name"]) % 5) / 10,  # Random rating 4.5-5.0
                is_active=True,
                is_new=False
            )
            session.add(food)
        
        await session.commit()
        logger.info("Default data created")

# ====================== FSM STATES ======================

class AdminStates(StatesGroup):
    # Food management
    waiting_food_action = State()
    waiting_food_name = State()
    waiting_food_category = State()
    waiting_food_price = State()
    waiting_food_description = State()
    waiting_food_rating = State()
    waiting_food_image = State()
    
    # Promo management
    waiting_promo_code = State()
    waiting_promo_discount = State()
    waiting_promo_expires = State()
    waiting_promo_limit = State()
    
    # Courier management
    waiting_courier_chat_id = State()
    waiting_courier_name = State()

# ====================== FASTAPI APP ======================

app = FastAPI(title="FIESTA Food Delivery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OrderCreateRequest(BaseModel):
    items: List[Dict[str, Any]]
    total: float
    customer_name: str
    phone: str
    comment: Optional[str] = ""
    location: Dict[str, float]
    promo_code: Optional[str] = None

def verify_telegram_webapp_data(init_data: str, bot_token: str) -> Optional[Dict]:
    """Verify Telegram WebApp initData"""
    try:
        parsed_data = dict(x.split('=', 1) for x in unquote(init_data).split('&'))
        
        data_check_string_parts = []
        for key in sorted(parsed_data.keys()):
            if key != 'hash':
                data_check_string_parts.append(f"{key}={parsed_data[key]}")
        
        data_check_string = '\n'.join(data_check_string_parts)
        
        secret_key = hmac.new("WebAppData".encode(), bot_token.encode(), hashlib.sha256).digest()
        calculated_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
        
        if calculated_hash == parsed_data.get('hash'):
            return parsed_data
        return None
    except Exception as e:
        logger.error(f"Telegram data verification error: {e}")
        return None

@app.get("/")
async def root():
    return {"status": "ok", "service": "FIESTA Food Delivery API"}

@app.get("/api/categories")
async def get_categories(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Category).where(Category.is_active == True))
    categories = result.scalars().all()
    return [{"id": c.id, "name": c.name} for c in categories]

@app.get("/api/foods")
async def get_foods(category_id: Optional[int] = None, session: AsyncSession = Depends(get_session)):
    query = select(Food).where(Food.is_active == True)
    if category_id:
        query = query.where(Food.category_id == category_id)
    
    result = await session.execute(query)
    foods = result.scalars().all()
    
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
async def validate_promo(code: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Promo).where(
            Promo.code == code.upper(),
            Promo.is_active == True
        )
    )
    promo = result.scalar_one_or_none()
    
    if not promo:
        raise HTTPException(status_code=404, detail="Promo code not found")
    
    if promo.expires_at and promo.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Promo code expired")
    
    if promo.usage_limit and promo.used_count >= promo.usage_limit:
        raise HTTPException(status_code=400, detail="Promo code limit reached")
    
    return {"discount_percent": promo.discount_percent}

# ====================== BOT SETUP ======================

bot = Bot(token=config.BOT_TOKEN)
redis_storage = Redis.from_url(config.REDIS_URL)
storage = RedisStorage(redis_storage)
dp = Dispatcher(storage=storage)

# ====================== HELPER FUNCTIONS ======================

async def get_or_create_user(tg_id: int, username: Optional[str], full_name: str, ref_by_user_id: Optional[int] = None) -> User:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.tg_id == tg_id))
        user = result.scalar_one_or_none()
        
        if not user:
            user = User(
                tg_id=tg_id,
                username=username,
                full_name=full_name,
                ref_by_user_id=ref_by_user_id
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info(f"New user created: {tg_id} - {full_name}")
        
        return user

async def create_order(user_id: int, order_data: OrderCreateRequest) -> Order:
    async with AsyncSessionLocal() as session:
        # Generate order number
        order_number = f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Apply promo if exists
        total = order_data.total
        if order_data.promo_code:
            result = await session.execute(
                select(Promo).where(
                    Promo.code == order_data.promo_code.upper(),
                    Promo.is_active == True
                )
            )
            promo = result.scalar_one_or_none()
            if promo:
                discount = total * (promo.discount_percent / 100)
                total -= discount
                promo.used_count += 1
        
        # Create order
        order = Order(
            order_number=order_number,
            user_id=user_id,
            customer_name=order_data.customer_name,
            phone=order_data.phone,
            comment=order_data.comment or "",
            total=total,
            status="NEW",
            location_lat=order_data.location["lat"],
            location_lng=order_data.location["lng"]
        )
        session.add(order)
        await session.flush()
        
        # Create order items
        for item in order_data.items:
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

async def get_user_orders(tg_id: int, limit: int = 10) -> List[Order]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.tg_id == tg_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return []
        
        result = await session.execute(
            select(Order).where(Order.user_id == user.id).order_by(Order.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

async def get_referral_stats(tg_id: int) -> Dict:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.tg_id == tg_id))
        user = result.scalar_one_or_none()
        
        if not user:
            return {"ref_count": 0, "orders_count": 0, "paid_count": 0}
        
        # Count referrals
        result = await session.execute(
            select(func.count(User.id)).where(User.ref_by_user_id == user.id)
        )
        ref_count = result.scalar() or 0
        
        # Count orders from referrals
        result = await session.execute(
            select(func.count(Order.id)).select_from(Order).join(User).where(User.ref_by_user_id == user.id)
        )
        orders_count = result.scalar() or 0
        
        # Count delivered orders from referrals
        result = await session.execute(
            select(func.count(Order.id)).select_from(Order).join(User).where(
                User.ref_by_user_id == user.id,
                Order.status == "DELIVERED"
            )
        )
        paid_count = result.scalar() or 0
        
        return {
            "ref_count": ref_count,
            "orders_count": orders_count,
            "paid_count": paid_count,
            "promo_given": user.promo_given
        }

async def create_referral_promo(tg_id: int) -> Optional[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.tg_id == tg_id))
        user = result.scalar_one_or_none()
        
        if not user or user.promo_given:
            return None
        
        # Create promo code
        promo_code = f"REF{user.id}{datetime.now().strftime('%m%d')}"
        promo = Promo(
            code=promo_code,
            discount_percent=15,
            expires_at=datetime.utcnow() + timedelta(days=30),
            usage_limit=1,
            is_active=True
        )
        session.add(promo)
        
        user.promo_given = True
        await session.commit()
        
        return promo_code

async def send_to_admin_channel(order: Order):
    """Send order to admin channel"""
    async with AsyncSessionLocal() as session:
        # Refresh order with items
        await session.refresh(order, ["items", "user"])
        
        items_text = "\n".join([
            f"🍽️ {item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум"
            for item in order.items
        ])
        
        maps_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
        
        text = f"""🆕 <b>Новый заказ №{order.order_number}</b>

👤 Пользователь: {order.customer_name}
👤 Username: @{order.user.username or 'не указан'}
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}
📍 <a href="{maps_link}">Локация на карте</a>

📝 Заказ:
{items_text}

💬 Комментарий: {order.comment or 'нет'}
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"order_status:CONFIRMED:{order.id}"),
                InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order_status:COOKING:{order.id}")
            ],
            [
                InlineKeyboardButton(text="🚴 Курьер", callback_data=f"select_courier:{order.id}")
            ],
            [
                InlineKeyboardButton(text="❌ Отменить", callback_data=f"order_status:CANCELED:{order.id}")
            ]
        ])
        
        try:
            msg = await bot.send_message(
                config.SHOP_CHANNEL_ID,
                text,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
            
            # Save message ID
            await session.execute(
                sql_update(Order).where(Order.id == order.id).values(admin_message_id=msg.message_id)
            )
            await session.commit()
        except Exception as e:
            logger.error(f"Error sending to admin channel: {e}")

async def update_admin_channel_message(order_id: int, new_status: str):
    """Update admin channel message with new status"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if not order or not order.admin_message_id:
            return
        
        await session.refresh(order, ["items", "user"])
        
        items_text = "\n".join([
            f"🍽️ {item.name_snapshot} x{item.qty} = {item.line_total:,.0f} сум"
            for item in order.items
        ])
        
        maps_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
        
        status_emoji = {
            "NEW": "🆕",
            "CONFIRMED": "✅",
            "COOKING": "🍳",
            "COURIER_ASSIGNED": "🚴",
            "OUT_FOR_DELIVERY": "📦",
            "DELIVERED": "✅",
            "CANCELED": "❌"
        }
        
        text = f"""{status_emoji.get(new_status, '📦')} <b>Заказ №{order.order_number}</b>
<b>Статус: {get_status_name(new_status)}</b>

👤 Пользователь: {order.customer_name}
👤 Username: @{order.user.username or 'не указан'}
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}
📍 <a href="{maps_link}">Локация на карте</a>

📝 Заказ:
{items_text}

💬 Комментарий: {order.comment or 'нет'}
"""
        
        # Update keyboard based on status
        if new_status in ["DELIVERED", "CANCELED"]:
            keyboard = None
        elif new_status == "COURIER_ASSIGNED":
            keyboard = InlineKeyboardMarkup(inline_keyboard=[])
        else:
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"order_status:CONFIRMED:{order.id}"),
                    InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order_status:COOKING:{order.id}")
                ],
                [
                    InlineKeyboardButton(text="🚴 Курьер", callback_data=f"select_courier:{order.id}")
                ]
            ])
        
        try:
            await bot.edit_message_text(
                text,
                config.SHOP_CHANNEL_ID,
                order.admin_message_id,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Error updating admin message: {e}")

def get_status_name(status: str) -> str:
    status_names = {
        "NEW": "Принят",
        "CONFIRMED": "Подтвержден",
        "COOKING": "Готовится",
        "COURIER_ASSIGNED": "Курьер назначен",
        "OUT_FOR_DELIVERY": "Передан курьеру",
        "DELIVERED": "Доставлен",
        "CANCELED": "Отменен"
    }
    return status_names.get(status, status)

# ====================== CLIENT HANDLERS ======================

def get_main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")]
        ],
        resize_keyboard=True
    )

@dp.message(Command("start"))
async def cmd_start(message: Message):
    # Handle referral
    ref_user_id = None
    if message.text and len(message.text.split()) > 1:
        try:
            ref_user_id = int(message.text.split()[1])
            if ref_user_id == message.from_user.id:
                ref_user_id = None
        except:
            pass
    
    # Get or create user
    user = await get_or_create_user(
        message.from_user.id,
        message.from_user.username,
        message.from_user.full_name,
        ref_user_id
    )
    
    await message.answer(
        f"Добро пожаловать в FIESTA! {message.from_user.full_name}\n\n"
        f"Для заказа перейдите по кнопке ➡️ 🛍 Заказать",
        reply_markup=get_main_keyboard()
    )

@dp.message(Command("shop"))
async def cmd_shop(message: Message):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))]
    ])
    
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=keyboard
    )

@dp.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    orders = await get_user_orders(message.from_user.id, limit=10)
    
    if not orders:
        await message.answer(
            "В данный момент у вас нет активных заказов в нашем магазине.\n"
            "Чтобы открыть магазин, введите команду — /shop"
        )
        return
    
    async with AsyncSessionLocal() as session:
        for order in orders:
            await session.refresh(order, ["items"])
            
            items_text = "\n".join([
                f"  • {item.name_snapshot} x{item.qty} - {item.line_total:,.0f} сум"
                for item in order.items
            ])
            
            text = f"""🆔 Заказ №{order.order_number}
📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}
💰 {order.total:,.0f} сум
📦 Статус: {get_status_name(order.status)}

📝 Состав заказа:
{items_text}
"""
            await message.answer(text)

@dp.message(F.text == "ℹ️ Информация о нас")
async def info_about_us(message: Message):
    text = """🌟 Добро Пожаловать в FIESTA!

📍 Наш адрес: Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи
🏢 Ориентир: Школа №12 Оруджева
📞 Контактный номер: +998 91 420 15 15
🕙 Рабочие часы: 24/7

📷 Мы в Instagram: <a href="https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo">fiesta.khiva</a>
🔗 Найти нас на карте: <a href="https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7">Место расположение</a>
"""
    
    await message.answer(text, parse_mode="HTML", disable_web_page_preview=True)

@dp.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    stats = await get_referral_stats(message.from_user.id)
    
    ref_link = f"https://t.me/{config.BOT_USERNAME}?start={message.from_user.id}"
    
    text = f"""За приглашение друга, вы можете получить промо-код от нас

👥 Вы пригласили {stats['ref_count']} человек
🛒 Оформили заказов: {stats['orders_count']}
💰 Оплатили заказов: {stats['paid_count']}

👤 Ваша реферальная ссылка:
{ref_link}

Пригласите трех человек и вы получите от нас промо-код со скидкой 15%
"""
    
    # Give promo if eligible
    if stats['ref_count'] >= 3 and not stats['promo_given']:
        promo_code = await create_referral_promo(message.from_user.id)
        if promo_code:
            text += f"\n\n🎉 Поздравляем! Вы получили промо-код: <b>{promo_code}</b>\nСкидка 15% на следующий заказ!"
    
    await message.answer(text, parse_mode="HTML")

@dp.message(F.web_app_data)
async def handle_webapp_data(message: Message):
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get("type") == "order_create":
            # Validate total
            if data["total"] < 50000:
                await message.answer("❌ Минимальная сумма заказа 50,000 сум")
                return
            
            # Get user
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(User).where(User.tg_id == message.from_user.id))
                user = result.scalar_one_or_none()
                
                if not user:
                    await message.answer("❌ Ошибка: пользователь не найден")
                    return
                
                # Create order
                order_req = OrderCreateRequest(
                    items=data["items"],
                    total=data["total"],
                    customer_name=data["customer_name"],
                    phone=data["phone"],
                    comment=data.get("comment", ""),
                    location=data["location"],
                    promo_code=data.get("promo_code")
                )
                
                order = await create_order(user.id, order_req)
                
                # Notify user
                await message.answer(
                    f"Ваш заказ принят ✅\n\n"
                    f"🆔 Заказ №{order.order_number}\n"
                    f"💰 Сумма: {order.total:,.0f} сум\n"
                    f"📦 Статус: Принят"
                )
                
                # Send to admin channel
                await send_to_admin_channel(order)
                
    except Exception as e:
        logger.error(f"Error handling webapp data: {e}")
        await message.answer("❌ Произошла ошибка при обработке заказа")

# ====================== ADMIN HANDLERS ======================

def is_admin(user_id: int) -> bool:
    return user_id in config.ADMIN_IDS

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("❌ У вас нет доступа к админ-панели")
        return
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Таомлар", callback_data="admin_foods")],
        [InlineKeyboardButton(text="📂 Категориялар", callback_data="admin_categories")],
        [InlineKeyboardButton(text="🎁 Промокодлар", callback_data="admin_promos")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🚴 Курьерлар", callback_data="admin_couriers")],
        [InlineKeyboardButton(text="📦 Актив буюртмалар", callback_data="admin_active_orders")]
    ])
    
    await message.answer("👨‍💼 <b>Админ панель</b>", reply_markup=keyboard, parse_mode="HTML")

@dp.callback_query(F.data == "admin_stats")
async def admin_statistics(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    async with AsyncSessionLocal() as session:
        # Today stats
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        result = await session.execute(
            select(func.count(Order.id)).where(Order.created_at >= today)
        )
        today_orders = result.scalar() or 0
        
        result = await session.execute(
            select(func.count(Order.id)).where(
                Order.created_at >= today,
                Order.status == "DELIVERED"
            )
        )
        today_delivered = result.scalar() or 0
        
        result = await session.execute(
            select(func.sum(Order.total)).where(
                Order.created_at >= today,
                Order.status == "DELIVERED"
            )
        )
        today_revenue = result.scalar() or 0
        
        # Active orders
        result = await session.execute(
            select(func.count(Order.id)).where(
                Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
            )
        )
        active_orders = result.scalar() or 0
        
        text = f"""📊 <b>Статистика</b>

<b>Сегодня:</b>
📦 Заказов: {today_orders}
✅ Доставлено: {today_delivered}
💰 Выручка: {today_revenue:,.0f} сум

📦 Активных заказов: {active_orders}
"""
        
        await callback.message.edit_text(text, parse_mode="HTML")
    
    await callback.answer()

@dp.callback_query(F.data == "admin_active_orders")
async def admin_active_orders(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order).where(
                Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
            ).order_by(Order.created_at.desc())
        )
        orders = result.scalars().all()
        
        if not orders:
            await callback.message.edit_text("📦 Нет активных заказов")
            await callback.answer()
            return
        
        text = "📦 <b>Активные заказы:</b>\n\n"
        
        for order in orders[:10]:
            text += f"🆔 #{order.order_number} | {get_status_name(order.status)}\n"
            text += f"💰 {order.total:,.0f} сум | {order.created_at.strftime('%d.%m %H:%M')}\n\n"
        
        await callback.message.edit_text(text, parse_mode="HTML")
    
    await callback.answer()

@dp.callback_query(F.data.startswith("order_status:"))
async def change_order_status(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    _, new_status, order_id = callback.data.split(":")
    order_id = int(order_id)
    
    async with AsyncSessionLocal() as session:
        # Update order status
        await session.execute(
            sql_update(Order).where(Order.id == order_id).values(
                status=new_status,
                updated_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        # Get order details
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one()
        
        # Notify user
        try:
            await bot.send_message(
                order.user.tg_id,
                f"Статус вашего заказа №{order.order_number} изменен:\n"
                f"📦 {get_status_name(new_status)}"
            )
        except Exception as e:
            logger.error(f"Error notifying user: {e}")
        
        # Update admin channel message
        await update_admin_channel_message(order_id, new_status)
    
    await callback.answer(f"✅ Статус изменен на: {get_status_name(new_status)}")

@dp.callback_query(F.data.startswith("select_courier:"))
async def select_courier(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Courier).where(Courier.is_active == True)
        )
        couriers = result.scalars().all()
        
        if not couriers:
            await callback.answer("❌ Нет доступных курьеров")
            return
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=f"🚴 {c.name}", callback_data=f"assign_courier:{order_id}:{c.id}")]
            for c in couriers
        ])
        
        await callback.message.edit_text(
            f"Выберите курьера для заказа:",
            reply_markup=keyboard
        )
    
    await callback.answer()

@dp.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    async with AsyncSessionLocal() as session:
        # Update order
        await session.execute(
            sql_update(Order).where(Order.id == order_id).values(
                status="COURIER_ASSIGNED",
                courier_id=courier_id,
                updated_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        # Get order and courier
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one()
        await session.refresh(order, ["items", "courier", "user"])
        
        # Send to courier
        items_text = "\n".join([
            f"🍽️ {item.name_snapshot} x{item.qty}"
            for item in order.items
        ])
        
        maps_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
        
        courier_text = f"""🚴 <b>Новый заказ №{order.order_number}</b>

👤 Клиент: {order.customer_name}
📞 Телефон: {order.phone}
💰 Сумма: {order.total:,.0f} сум
📍 <a href="{maps_link}">Локация на карте</a>

🍽️ Список:
{items_text}

💬 Комментарий: {order.comment or 'нет'}
"""
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="✅ Қабул қилдим", callback_data=f"courier_accept:{order.id}")],
            [InlineKeyboardButton(text="📦 Етказилди", callback_data=f"courier_delivered:{order.id}")]
        ])
        
        try:
            await bot.send_message(
                order.courier.chat_id,
                courier_text,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
            
            # Also send to courier channel if configured
            await bot.send_message(
                config.COURIER_CHANNEL_ID,
                courier_text,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
        except Exception as e:
            logger.error(f"Error sending to courier: {e}")
        
        # Notify user
        try:
            await bot.send_message(
                order.user.tg_id,
                f"Ваш заказ №{order.order_number} передан курьеру 🚴"
            )
        except Exception as e:
            logger.error(f"Error notifying user: {e}")
        
        # Update admin message
        await update_admin_channel_message(order_id, "COURIER_ASSIGNED")
    
    await callback.answer("✅ Курьер назначен")

# ====================== COURIER HANDLERS ======================

@dp.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Check if courier is assigned
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if not order:
            await callback.answer("❌ Заказ не найден")
            return
        
        if order.courier_id is None:
            await callback.answer("❌ Вы не назначены на этот заказ")
            return
        
        result = await session.execute(select(Courier).where(Courier.id == order.courier_id))
        courier = result.scalar_one_or_none()
        
        if not courier or courier.chat_id != callback.from_user.id:
            await callback.answer("❌ Вы не назначены на этот заказ")
            return
        
        # Update status
        await session.execute(
            sql_update(Order).where(Order.id == order_id).values(
                status="OUT_FOR_DELIVERY",
                updated_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        # Notify user
        await session.refresh(order, ["user"])
        try:
            await bot.send_message(
                order.user.tg_id,
                f"Ваш заказ №{order.order_number} передан курьеру 🚴"
            )
        except Exception as e:
            logger.error(f"Error notifying user: {e}")
        
        # Update admin message
        await update_admin_channel_message(order_id, "OUT_FOR_DELIVERY")
    
    await callback.answer("✅ Заказ принят")
    await callback.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📦 Етказилди", callback_data=f"courier_delivered:{order.id}")]
    ]))

@dp.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered(callback: CallbackQuery):
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Check if courier is assigned
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if not order:
            await callback.answer("❌ Заказ не найден")
            return
        
        if order.courier_id is None:
            await callback.answer("❌ Вы не назначены на этот заказ")
            return
        
        result = await session.execute(select(Courier).where(Courier.id == order.courier_id))
        courier = result.scalar_one_or_none()
        
        if not courier or courier.chat_id != callback.from_user.id:
            await callback.answer("❌ Вы не назначены на этот заказ")
            return
        
        # Update status
        await session.execute(
            sql_update(Order).where(Order.id == order_id).values(
                status="DELIVERED",
                delivered_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await session.commit()
        
        # Notify user
        await session.refresh(order, ["user"])
        try:
            await bot.send_message(
                order.user.tg_id,
                f"Ваш заказ №{order.order_number} успешно доставлен 🎉\nСпасибо!"
            )
        except Exception as e:
            logger.error(f"Error notifying user: {e}")
        
        # Update admin message
        await update_admin_channel_message(order_id, "DELIVERED")
    
    await callback.answer("✅ Заказ доставлен")
    await callback.message.edit_reply_markup(reply_markup=None)

@dp.callback_query(F.data == "admin_couriers")
async def admin_couriers_list(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Courier))
        couriers = result.scalars().all()
        
        if not couriers:
            text = "🚴 Список курьеров пуст"
        else:
            text = "🚴 <b>Курьеры:</b>\n\n"
            for c in couriers:
                status = "✅ Активен" if c.is_active else "❌ Неактивен"
                text += f"ID: {c.id} | {c.name} | {status}\n"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Добавить курьера", callback_data="admin_add_courier")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    
    await callback.answer()

@dp.callback_query(F.data == "admin_add_courier")
async def admin_add_courier_start(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    await callback.message.edit_text("Введите chat_id курьера:")
    await state.set_state(AdminStates.waiting_courier_chat_id)
    await callback.answer()

@dp.message(AdminStates.waiting_courier_chat_id)
async def admin_add_courier_chat_id(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    
    try:
        chat_id = int(message.text)
        await state.update_data(courier_chat_id=chat_id)
        await message.answer("Введите имя курьера:")
        await state.set_state(AdminStates.waiting_courier_name)
    except ValueError:
        await message.answer("❌ Неверный формат. Введите числовой chat_id:")

@dp.message(AdminStates.waiting_courier_name)
async def admin_add_courier_name(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    
    data = await state.get_data()
    
    async with AsyncSessionLocal() as session:
        courier = Courier(
            chat_id=data["courier_chat_id"],
            name=message.text,
            is_active=True
        )
        session.add(courier)
        await session.commit()
    
    await message.answer("✅ Курьер добавлен успешно!")
    await state.clear()

@dp.callback_query(F.data == "admin_promos")
async def admin_promos_list(callback: CallbackQuery):
    if not is_admin(callback.from_user.id):
        await callback.answer("❌ Доступ запрещен")
        return
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Promo).where(Promo.is_active == True))
        promos = result.scalars().all()
        
        if not promos:
            text = "🎁 Список промокодов пуст"
        else:
            text = "🎁 <b>Промокоды:</b>\n\n"
            for p in promos:
                expires = p.expires_at.strftime('%d.%m.%Y') if p.expires_at else "Без срока"
                text += f"<b>{p.code}</b> | -{p.discount_percent}%\n"
                text += f"Использовано: {p.used_count}/{p.usage_limit or '∞'} | До: {expires}\n\n"
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="➕ Создать промокод", callback_data="admin_create_promo")],
            [InlineKeyboardButton(text="◀️ Назад", callback_data="admin_back")]
        ])
        
        await callback.message.edit_text(text, reply_markup=keyboard, parse_mode="HTML")
    
    await callback.answer()

@dp.callback_query(F.data == "admin_back")
async def admin_back(callback: CallbackQuery):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Таомлар", callback_data="admin_foods")],
        [InlineKeyboardButton(text="📂 Категориялар", callback_data="admin_categories")],
        [InlineKeyboardButton(text="🎁 Промокодлар", callback_data="admin_promos")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🚴 Курьерлар", callback_data="admin_couriers")],
        [InlineKeyboardButton(text="📦 Актив буюртмалар", callback_data="admin_active_orders")]
    ])
    
    await callback.message.edit_text("👨‍💼 <b>Админ панель</b>", reply_markup=keyboard, parse_mode="HTML")
    await callback.answer()

# ====================== MAIN ======================

async def start_bot():
    # Initialize database
    await init_db()
    
    # Get bot info
    bot_info = await bot.get_me()
    config.BOT_USERNAME = bot_info.username
    logger.info(f"Bot started: @{config.BOT_USERNAME}")
    
    # Start polling
    await dp.start_polling(bot)

async def start_api():
    config_uvicorn = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config_uvicorn)
    await server.serve()

async def main():
    # Run both bot and API
    await asyncio.gather(
        start_bot(),
        start_api()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
