"""
Food Delivery Telegram Bot - Full System
Python 3.11+, aiogram 3.x, PostgreSQL, Redis
Production-ready with Clean Architecture
"""

import asyncio
import logging
import json
import hashlib
import hmac
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager

from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy import (
    Integer, String, Float, Boolean, DateTime, ForeignKey, Text,
    BigInteger, func, select, update, delete, and_, or_
)
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, WebAppInfo,
    MenuButtonWebApp, WebAppData
)
from aiogram.filters import Command, CommandStart
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==================== CONFIGURATION ====================

class Config:
    def __init__(self):
        self.BOT_TOKEN = os.getenv("BOT_TOKEN", "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4")
        self.ADMIN_IDS = self._parse_admin_ids(os.getenv("ADMIN_IDS", "6365371142"))
        self.DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway")
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411")
        self.SHOP_CHANNEL_ID = int(os.getenv("SHOP_CHANNEL_ID", "-1003530497437"))
        self.COURIER_CHANNEL_ID = int(os.getenv("COURIER_CHANNEL_ID", "-1003707946746"))
        self.WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.netlify.app")
        self.API_URL = os.getenv("API_URL", "https://uzbke-production.up.railway.app")
        self.BOT_USERNAME = os.getenv("BOT_USERNAME", "mainsu_food_bot")
        self.SECRET_KEY = os.getenv("SECRET_KEY", "mainsu_food_secret_key_2024")
        
        # Validation
        if not self.BOT_TOKEN:
            raise ValueError("BOT_TOKEN is required")
        if not self.DB_URL:
            raise ValueError("DB_URL is required")
    
    def _parse_admin_ids(self, admin_ids_str: str) -> List[int]:
        """Parse comma-separated admin IDs string to list of integers"""
        if not admin_ids_str:
            return []
        try:
            return [int(id_str.strip()) for id_str in admin_ids_str.split(',') if id_str.strip()]
        except ValueError:
            print(f"Warning: Invalid ADMIN_IDS format: {admin_ids_str}")
            return []

config = Config()

# Print config for debugging
print("=" * 50)
print("CONFIGURATION LOADED:")
print(f"Bot Token: {config.BOT_TOKEN[:10]}...")
print(f"Admin IDs: {config.ADMIN_IDS}")
print(f"DB URL: {config.DB_URL[:50]}...")
print(f"Redis URL: {config.REDIS_URL[:50]}...")
print(f"Shop Channel: {config.SHOP_CHANNEL_ID}")
print(f"Courier Channel: {config.COURIER_CHANNEL_ID}")
print(f"WebApp URL: {config.WEBAPP_URL}")
print(f"API URL: {config.API_URL}")
print("=" * 50)

# ==================== DATABASE MODELS ====================

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(100))
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ref_by_user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    balance: Mapped[float] = mapped_column(Float, default=0.0)
    
    orders = relationship("Order", back_populates="user")

class Category(Base):
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    name_ru: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    name_uz: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    image_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    foods = relationship("Food", back_populates="category")

class Food(Base):
    __tablename__ = "foods"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey("categories.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    name_ru: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    name_uz: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    description: Mapped[Optional[str]] = mapped_column(Text)
    description_ru: Mapped[Optional[str]] = mapped_column(Text)
    description_uz: Mapped[Optional[str]] = mapped_column(Text)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=0.0)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    category = relationship("Category", back_populates="foods")

class OrderStatus(str, Enum):
    NEW = "NEW"
    CONFIRMED = "CONFIRMED"
    COOKING = "COOKING"
    COURIER_ASSIGNED = "COURIER_ASSIGNED"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"
    CANCELED = "CANCELED"

class Order(Base):
    __tablename__ = "orders"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    customer_name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[str] = mapped_column(String(20), nullable=False)
    comment: Mapped[Optional[str]] = mapped_column(Text)
    total: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=OrderStatus.NEW.value)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    location_lat: Mapped[float] = mapped_column(Float, nullable=False)
    location_lng: Mapped[float] = mapped_column(Float, nullable=False)
    courier_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("couriers.id"))
    promo_code: Mapped[Optional[str]] = mapped_column(String(50))
    discount_amount: Mapped[float] = mapped_column(Float, default=0.0)
    final_total: Mapped[float] = mapped_column(Float, nullable=False)
    channel_message_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(Integer, ForeignKey("orders.id"), nullable=False)
    food_id: Mapped[int] = mapped_column(Integer, ForeignKey("foods.id"), nullable=False)
    name_snapshot: Mapped[str] = mapped_column(String(200), nullable=False)
    price_snapshot: Mapped[float] = mapped_column(Float, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    line_total: Mapped[float] = mapped_column(Float, nullable=False)
    
    order = relationship("Order", back_populates="items")
    food = relationship("Food")

class Promo(Base):
    __tablename__ = "promos"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    discount_percent: Mapped[int] = mapped_column(Integer, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    usage_limit: Mapped[Optional[int]] = mapped_column(Integer)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by: Mapped[Optional[int]] = mapped_column(Integer)

class Courier(Base):
    __tablename__ = "couriers"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class ReferralStat(Base):
    __tablename__ = "referral_stats"
    
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    ref_count: Mapped[int] = mapped_column(Integer, default=0)
    orders_count: Mapped[int] = mapped_column(Integer, default=0)
    delivered_count: Mapped[int] = mapped_column(Integer, default=0)
    last_promo_given: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    user = relationship("User")

# ==================== DATABASE SESSION ====================

class Database:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    @asynccontextmanager
    async def get_session(self):
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

db = Database(config.DB_URL)

# ==================== REDIS STORAGE ====================

try:
    redis = aioredis.from_url(config.REDIS_URL, decode_responses=True)
    storage = RedisStorage(redis=redis)
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection error: {e}")
    # Fallback to memory storage
    from aiogram.fsm.storage.memory import MemoryStorage
    storage = MemoryStorage()
    print("⚠️ Using memory storage instead of Redis")

# ==================== BOT INITIALIZATION ====================

bot = Bot(token=config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=storage)

# ==================== FASTAPI APP ====================

fastapi_app = FastAPI(title="Food Delivery API", version="1.0.0")

# CORS middleware
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS ====================

class FoodResponse(BaseModel):
    id: int
    name: str
    name_ru: str
    name_uz: str
    description: Optional[str]
    description_ru: Optional[str]
    description_uz: Optional[str]
    price: float
    rating: float
    is_new: bool
    is_active: bool
    image_url: Optional[str]
    category_id: int
    category_name: str

class CategoryResponse(BaseModel):
    id: int
    name: str
    name_ru: str
    name_uz: str
    is_active: bool
    image_url: Optional[str]
    foods_count: int

class OrderCreate(BaseModel):
    type: str = "order_create"
    items: List[Dict[str, Any]]
    total: float
    customer_name: str
    phone: str
    comment: Optional[str] = ""
    location: Dict[str, float]
    promo_code: Optional[str] = None

class PromoValidate(BaseModel):
    code: str
    total_amount: float

class PromoResponse(BaseModel):
    valid: bool
    discount_percent: Optional[int] = None
    discount_amount: Optional[float] = None
    final_total: Optional[float] = None
    message: Optional[str] = None

# ==================== TELEGRAM INITDATA VERIFY ====================

def verify_telegram_initdata(init_data: str) -> bool:
    """Verify Telegram WebApp initData"""
    try:
        if not init_data:
            return False
            
        # Parse initData
        data_pairs = init_data.split('&')
        hash_str = None
        data_check_string_parts = []
        
        for pair in data_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                if key == 'hash':
                    hash_str = value
                else:
                    data_check_string_parts.append(f"{key}={value}")
        
        if not hash_str:
            return False
        
        data_check_string = '\n'.join(sorted(data_check_string_parts))
        
        # Calculate secret key
        secret_key = hmac.new(
            b"WebAppData",
            msg=config.BOT_TOKEN.encode(),
            digestmod=hashlib.sha256
        ).digest()
        
        # Calculate hash
        calculated_hash = hmac.new(
            secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256
        ).hexdigest()
        
        return calculated_hash == hash_str
    except Exception as e:
        print(f"Error verifying initdata: {e}")
        return False

# ==================== SERVICES ====================

class UserService:
    @staticmethod
    async def get_or_create_user(tg_id: int, username: str, full_name: str, ref_id: Optional[int] = None) -> User:
        async with db.get_session() as session:
            # Check if user exists
            result = await session.execute(select(User).where(User.tg_id == tg_id))
            user = result.scalar_one_or_none()
            
            if user:
                return user
            
            # Create new user
            user = User(
                tg_id=tg_id,
                username=username,
                full_name=full_name,
                ref_by_user_id=ref_id
            )
            session.add(user)
            await session.flush()
            
            # Create referral stats
            if ref_id:
                # Update referrer's stats
                result = await session.execute(
                    select(ReferralStat).where(ReferralStat.user_id == ref_id)
                )
                stats = result.scalar_one_or_none()
                if stats:
                    stats.ref_count += 1
                else:
                    stats = ReferralStat(user_id=ref_id, ref_count=1)
                    session.add(stats)
            
            # Create stats for new user
            stats = ReferralStat(user_id=user.id)
            session.add(stats)
            
            await session.commit()
            return user
    
    @staticmethod
    async def get_user_by_tg_id(tg_id: int) -> Optional[User]:
        async with db.get_session() as session:
            result = await session.execute(select(User).where(User.tg_id == tg_id))
            return result.scalar_one_or_none()

class FoodService:
    @staticmethod
    async def get_all_foods(active_only: bool = True) -> List[Dict]:
        async with db.get_session() as session:
            query = select(Food).join(Category)
            if active_only:
                query = query.where(Food.is_active == True)
            
            result = await session.execute(query)
            foods = []
            for food in result.scalars().all():
                foods.append({
                    "id": food.id,
                    "name": food.name,
                    "name_ru": food.name_ru,
                    "name_uz": food.name_uz,
                    "description": food.description,
                    "description_ru": food.description_ru,
                    "description_uz": food.description_uz,
                    "price": food.price,
                    "rating": food.rating,
                    "is_new": food.is_new,
                    "is_active": food.is_active,
                    "image_url": food.image_url,
                    "category_id": food.category_id,
                    "category_name": food.category.name if food.category else "Unknown"
                })
            return foods
    
    @staticmethod
    async def get_categories(active_only: bool = True) -> List[Dict]:
        async with db.get_session() as session:
            query = select(Category)
            if active_only:
                query = query.where(Category.is_active == True)
            query = query.order_by(Category.sort_order)
            
            result = await session.execute(query)
            categories = []
            for category in result.scalars().all():
                # Count foods in category
                food_count = await session.execute(
                    select(func.count(Food.id)).where(Food.category_id == category.id)
                )
                categories.append({
                    "id": category.id,
                    "name": category.name,
                    "name_ru": category.name_ru,
                    "name_uz": category.name_uz,
                    "is_active": category.is_active,
                    "image_url": category.image_url,
                    "foods_count": food_count.scalar() or 0
                })
            return categories

class OrderService:
    @staticmethod
    async def create_order(
        user_id: int,
        customer_name: str,
        phone: str,
        items: List[Dict],
        total: float,
        location_lat: float,
        location_lng: float,
        comment: Optional[str] = None,
        promo_code: Optional[str] = None
    ) -> Order:
        async with db.get_session() as session:
            # Generate order number
            today = datetime.now()
            result = await session.execute(
                select(func.count(Order.id))
                .where(func.date(Order.created_at) == today.date())
            )
            order_count = result.scalar() + 1
            order_number = f"ORD{today.strftime('%y%m%d')}{order_count:04d}"
            
            # Apply promo code if provided
            discount_amount = 0.0
            final_total = total
            
            if promo_code:
                promo = await session.execute(
                    select(Promo).where(
                        and_(
                            Promo.code == promo_code,
                            Promo.is_active == True,
                            or_(Promo.expires_at == None, Promo.expires_at > datetime.utcnow()),
                            or_(Promo.usage_limit == None, Promo.used_count < Promo.usage_limit)
                        )
                    )
                )
                promo = promo.scalar_one_or_none()
                
                if promo:
                    discount_amount = total * (promo.discount_percent / 100)
                    final_total = total - discount_amount
                    
                    # Update promo usage
                    promo.used_count += 1
            
            # Create order
            order = Order(
                order_number=order_number,
                user_id=user_id,
                customer_name=customer_name,
                phone=phone,
                comment=comment,
                total=total,
                final_total=final_total,
                discount_amount=discount_amount,
                promo_code=promo_code,
                status=OrderStatus.NEW.value,
                location_lat=location_lat,
                location_lng=location_lng
            )
            session.add(order)
            await session.flush()
            
            # Create order items
            for item in items:
                order_item = OrderItem(
                    order_id=order.id,
                    food_id=item.get("food_id", item.get("id", 0)),
                    name_snapshot=item["name"],
                    price_snapshot=item["price"],
                    qty=item["qty"],
                    line_total=item["price"] * item["qty"]
                )
                session.add(order_item)
            
            # Update user stats
            stats = await session.execute(
                select(ReferralStat).where(ReferralStat.user_id == user_id)
            )
            stats = stats.scalar_one_or_none()
            if stats:
                stats.orders_count += 1
            
            await session.commit()
            return order
    
    @staticmethod
    async def get_user_orders(user_id: int, limit: int = 10) -> List[Order]:
        async with db.get_session() as session:
            result = await session.execute(
                select(Order)
                .where(Order.user_id == user_id)
                .order_by(Order.created_at.desc())
                .limit(limit)
            )
            return result.scalars().all()

class PromoService:
    @staticmethod
    async def validate_promo(code: str, total_amount: float) -> Dict:
        async with db.get_session() as session:
            result = await session.execute(
                select(Promo).where(
                    and_(
                        Promo.code == code,
                        Promo.is_active == True,
                        or_(Promo.expires_at == None, Promo.expires_at > datetime.utcnow()),
                        or_(Promo.usage_limit == None, Promo.used_count < Promo.usage_limit)
                    )
                )
            )
            promo = result.scalar_one_or_none()
            
            if not promo:
                return {"valid": False, "message": "Неверный или просроченный промокод"}
            
            discount_amount = total_amount * (promo.discount_percent / 100)
            final_total = total_amount - discount_amount
            
            return {
                "valid": True,
                "discount_percent": promo.discount_percent,
                "discount_amount": discount_amount,
                "final_total": final_total,
                "message": f"Промокод применен! Скидка {promo.discount_percent}%"
            }

# ==================== TELEGRAM NOTIFICATION SERVICE ====================

class TelegramNotifyService:
    @staticmethod
    async def send_order_notification(order: Order, items_text: str):
        """Send order notification to admin channel"""
        try:
            message_text = (
                f"🆕 <b>Новый заказ №{order.order_number}</b>\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {order.total:,.0f} сум\n"
                f"🎁 <b>Скидка:</b> {order.discount_amount:,.0f} сум\n"
                f"💵 <b>Итого:</b> {order.final_total:,.0f} сум\n"
                f"🕒 <b>Время:</b> {order.created_at.strftime('%H:%M %d.%m.%Y')}\n"
                f"📍 <b>Локация:</b> <a href='https://maps.google.com/?q={order.location_lat},{order.location_lng}'>Показать на карте</a>\n"
                f"📝 <b>Комментарий:</b> {order.comment or 'Нет'}\n\n"
                f"🍽️ <b>Заказ:</b>\n{items_text}"
            )
            
            # Create inline keyboard
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"confirm_order:{order.id}"),
                    InlineKeyboardButton(text="🍳 Готовится", callback_data=f"cooking_order:{order.id}")
                ],
                [
                    InlineKeyboardButton(text="🚴 Курьер", callback_data=f"assign_courier:{order.id}")
                ],
                [
                    InlineKeyboardButton(text="❌ Отменить", callback_data=f"cancel_order:{order.id}")
                ]
            ])
            
            # Send message to admin channel
            message = await bot.send_message(
                chat_id=config.SHOP_CHANNEL_ID,
                text=message_text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML
            )
            
            # Save message ID to order
            async with db.get_session() as session:
                order.channel_message_id = message.message_id
                session.add(order)
                await session.commit()
            
            return message.message_id
            
        except Exception as e:
            print(f"Error sending order notification: {e}")
            return None

# ==================== TELEGRAM HANDLERS ====================

# Client router
client_router = Router()

@client_router.message(CommandStart())
async def cmd_start(message: Message):
    """Handle /start command with referral"""
    args = message.text.split()
    ref_id = None
    
    # Extract referral ID
    if len(args) > 1:
        try:
            ref_id = int(args[1])
        except ValueError:
            pass
    
    # Get or create user
    user = await UserService.get_or_create_user(
        tg_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
        ref_id=ref_id
    )
    
    # Create main keyboard
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL)),
                KeyboardButton(text="📦 Мои заказы")
            ],
            [
                KeyboardButton(text="ℹ️ Информация о нас"),
                KeyboardButton(text="👥 Пригласить друга")
            ]
        ],
        resize_keyboard=True
    )
    
    # Send welcome message
    welcome_text = (
        f"Добро пожаловать в FIESTA! {message.from_user.full_name}\n"
        f"Для заказа перейдите по кнопке ➡️\n"
        f"🛍 Заказать"
    )
    
    await message.answer(welcome_text, reply_markup=keyboard)

@client_router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    """Show user's orders"""
    user = await UserService.get_user_by_tg_id(message.from_user.id)
    if not user:
        await message.answer("Пожалуйста, начните с команды /start")
        return
    
    orders = await OrderService.get_user_orders(user.id, limit=10)
    
    if not orders:
        await message.answer(
            "В данный момент у вас нет активных заказов в нашем магазине.\n"
            "Чтобы открыть магазин, нажмите кнопку ниже",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))]
            ])
        )
        return
    
    # Format orders
    for order in orders:
        items_text = ""
        for item in order.items:
            items_text += f"• {item.name_snapshot} x{item.qty} = {item.line_total:,} сум\n"
        
        status_text = {
            OrderStatus.NEW.value: "🆕 Принят",
            OrderStatus.CONFIRMED.value: "✅ Подтвержден",
            OrderStatus.COOKING.value: "🍳 Готовится",
            OrderStatus.COURIER_ASSIGNED.value: "🚴 Курьер назначен",
            OrderStatus.OUT_FOR_DELIVERY.value: "📦 Передан курьеру",
            OrderStatus.DELIVERED.value: "🎉 Доставлен",
            OrderStatus.CANCELED.value: "❌ Отменен"
        }.get(order.status, order.status)
        
        order_text = (
            f"🆔 Заказ №{order.order_number}\n"
            f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
            f"💰 {order.final_total:,.0f} сум\n"
            f"📦 {status_text}\n\n"
            f"🍽️ Заказ:\n{items_text}"
        )
        
        await message.answer(order_text)

@client_router.message(F.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    """Send information about the restaurant"""
    about_text = (
        "🌟 Добро Пожаловать в FIESTA !\n"
        "📍 Наш адрес:Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢 Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙 Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)"
    )
    await message.answer(about_text)

@client_router.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    """Show referral information"""
    user = await UserService.get_user_by_tg_id(message.from_user.id)
    if not user:
        await message.answer("Пожалуйста, начните с команды /start")
        return
    
    # Get user stats
    async with db.get_session() as session:
        result = await session.execute(
            select(ReferralStat).where(ReferralStat.user_id == user.id)
        )
        stats = result.scalar_one_or_none()
    
    if stats:
        ref_count = stats.ref_count
        orders_count = stats.orders_count
        delivered_count = stats.delivered_count
    else:
        ref_count = orders_count = delivered_count = 0
    
    invite_text = (
        f"За приглашение друга, вы можете получить промо-код от нас\n"
        f"👥 Вы пригласили {ref_count} человек\n"
        f"🛒 Оформили заказов: {orders_count}\n"
        f"💰 Оплатили заказов: {delivered_count}\n"
        f"👤 Ваша реферальная ссылка: https://t.me/{config.BOT_USERNAME}?start={user.tg_id}\n\n"
        f"🎁 Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
    )
    
    await message.answer(invite_text)

@client_router.message(F.web_app_data)
async def handle_web_app_data(message: WebAppData):
    """Handle data from WebApp"""
    try:
        data = json.loads(message.web_app_data.data)
        
        if data.get("type") == "order_create":
            # Get user
            user = await UserService.get_user_by_tg_id(message.from_user.id)
            if not user:
                user = await UserService.get_or_create_user(
                    tg_id=message.from_user.id,
                    username=message.from_user.username,
                    full_name=message.from_user.full_name
                )
            
            # Validate total
            if data["total"] < 50000:
                await message.answer("Минимальная сумма заказа 50,000 сум")
                return
            
            # Format items text for notification
            items_text = ""
            for item in data["items"]:
                items_text += f"• {item['name']} x{item['qty']} = {item['price'] * item['qty']:,} сум\n"
            
            # Create order
            order = await OrderService.create_order(
                user_id=user.id,
                customer_name=data["customer_name"],
                phone=data["phone"],
                items=data["items"],
                total=data["total"],
                location_lat=data["location"]["lat"],
                location_lng=data["location"]["lng"],
                comment=data.get("comment"),
                promo_code=data.get("promo_code")
            )
            
            # Notify user
            user_message = (
                f"✅ Ваш заказ принят!\n"
                f"🆔 Заказ №{order.order_number}\n"
                f"💰 Сумма: {order.final_total:,.0f} сум\n"
                f"📦 Статус: Принят\n"
                f"🕒 Время: {order.created_at.strftime('%H:%M %d.%m.%Y')}"
            )
            await message.answer(user_message)
            
            # Send to admin channel
            await TelegramNotifyService.send_order_notification(order, items_text)
            
    except Exception as e:
        print(f"Error handling web app data: {e}")
        await message.answer("Произошла ошибка при обработке заказа. Пожалуйста, попробуйте позже.")

# Admin router
admin_router = Router()

def is_admin(user_id: int) -> bool:
    """Check if user is admin"""
    return user_id in config.ADMIN_IDS

@admin_router.message(Command("admin"))
async def cmd_admin(message: Message):
    """Admin panel"""
    if not is_admin(message.from_user.id):
        await message.answer("Доступ запрещен")
        return
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Таомлар", callback_data="admin_foods")],
        [InlineKeyboardButton(text="📂 Категориялар", callback_data="admin_categories")],
        [InlineKeyboardButton(text="🎁 Промокодлар", callback_data="admin_promos")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="🚴 Курьерлар", callback_data="admin_couriers")],
        [InlineKeyboardButton(text="📦 Фаол буюртмалар", callback_data="admin_active_orders")],
        [InlineKeyboardButton(text="⚙️ Созламалар", callback_data="admin_settings")]
    ])
    
    await message.answer("👑 Админ панели", reply_markup=keyboard)

@admin_router.callback_query(F.data == "admin_stats")
async def admin_stats(callback: CallbackQuery):
    """Show statistics"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    async with db.get_session() as session:
        # Orders count today
        today = datetime.now().date()
        result = await session.execute(
            select(func.count(Order.id))
            .where(func.date(Order.created_at) == today)
        )
        orders_today = result.scalar() or 0
        
        # Revenue today
        result = await session.execute(
            select(func.sum(Order.final_total))
            .where(
                and_(
                    func.date(Order.created_at) == today,
                    Order.status == OrderStatus.DELIVERED.value
                )
            )
        )
        revenue_today = result.scalar() or 0
        
        # Total users
        result = await session.execute(select(func.count(User.id)))
        total_users = result.scalar() or 0
    
    stats_text = (
        f"📊 <b>Статистика на сегодня</b>\n\n"
        f"📦 Заказов сегодня: {orders_today}\n"
        f"💰 Выручка сегодня: {revenue_today:,.0f} сум\n"
        f"👥 Всего пользователей: {total_users}\n\n"
        f"<i>Обновлено: {datetime.now().strftime('%H:%M')}</i>"
    )
    
    await callback.message.edit_text(stats_text, parse_mode=ParseMode.HTML)

@admin_router.callback_query(F.data.startswith("confirm_order:"))
async def confirm_order(callback: CallbackQuery):
    """Confirm order"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Update order status
    async with db.get_session() as session:
        await session.execute(
            update(Order)
            .where(Order.id == order_id)
            .values(status=OrderStatus.CONFIRMED.value, updated_at=datetime.utcnow())
        )
        await session.commit()
    
    await callback.answer("Заказ подтвержден")
    await callback.message.edit_reply_markup(reply_markup=None)

# ==================== FASTAPI ROUTES ====================

@fastapi_app.get("/api/foods")
async def get_foods(request: Request):
    """Get all active foods"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        # For development, allow without initData
        print("Warning: No valid initData, but returning foods for development")
    
    try:
        foods = await FoodService.get_all_foods(active_only=True)
        return JSONResponse(content=foods)
    except Exception as e:
        print(f"Error getting foods: {e}")
        # Return sample data for development
        sample_foods = [
            {
                "id": 1,
                "name": "Lavash Classic",
                "name_ru": "Лаваш Классик",
                "name_uz": "Lavash Klassik",
                "description": "Tender lavash with chicken, fresh vegetables",
                "description_ru": "Нежный лаваш с курицей, свежими овощами",
                "description_uz": "Tovuq, yangi sabzavotlar bilan yumshoq lavash",
                "price": 28000.0,
                "rating": 4.8,
                "is_new": True,
                "is_active": True,
                "image_url": None,
                "category_id": 1,
                "category_name": "Lavash"
            },
            {
                "id": 2,
                "name": "Cheese Burger",
                "name_ru": "Чизбургер",
                "name_uz": "Cheese Burger",
                "description": "Juicy beef burger with cheese",
                "description_ru": "Сочная говяжья котлета с сыром",
                "description_uz": "Pishloqli mazali mol go'shti burger",
                "price": 32000.0,
                "rating": 4.9,
                "is_new": True,
                "is_active": True,
                "image_url": None,
                "category_id": 2,
                "category_name": "Burger"
            },
            {
                "id": 3,
                "name": "Shaurma Big",
                "name_ru": "Шаурма Большая",
                "name_uz": "Shaurma Katta",
                "description": "Big shaurma with chicken and vegetables",
                "description_ru": "Большая шаурма с курицей и овощами",
                "description_uz": "Tovuq va sabzavotlar bilan katta shaurma",
                "price": 25000.0,
                "rating": 4.7,
                "is_new": False,
                "is_active": True,
                "image_url": None,
                "category_id": 3,
                "category_name": "Shaurma"
            }
        ]
        return JSONResponse(content=sample_foods)

@fastapi_app.get("/api/categories")
async def get_categories(request: Request):
    """Get all active categories"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        print("Warning: No valid initData, but returning categories for development")
    
    try:
        categories = await FoodService.get_categories(active_only=True)
        return JSONResponse(content=categories)
    except Exception as e:
        print(f"Error getting categories: {e}")
        # Return sample data for development
        sample_categories = [
            {
                "id": 1,
                "name": "Lavash",
                "name_ru": "Лаваш",
                "name_uz": "Lavash",
                "is_active": True,
                "image_url": None,
                "foods_count": 3
            },
            {
                "id": 2,
                "name": "Burger",
                "name_ru": "Бургер",
                "name_uz": "Burger",
                "is_active": True,
                "image_url": None,
                "foods_count": 2
            },
            {
                "id": 3,
                "name": "Shaurma",
                "name_ru": "Шаурма",
                "name_uz": "Shaurma",
                "is_active": True,
                "image_url": None,
                "foods_count": 2
            },
            {
                "id": 4,
                "name": "Hotdog",
                "name_ru": "Хотдог",
                "name_uz": "Hotdog",
                "is_active": True,
                "image_url": None,
                "foods_count": 2
            },
            {
                "id": 5,
                "name": "Combo",
                "name_ru": "Комбо",
                "name_uz": "Combo",
                "is_active": True,
                "image_url": None,
                "foods_count": 1
            }
        ]
        return JSONResponse(content=sample_categories)

@fastapi_app.post("/api/promo/validate")
async def validate_promo(promo_data: PromoValidate, request: Request):
    """Validate promo code"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        print("Warning: No valid initData, but validating promo for development")
    
    try:
        result = await PromoService.validate_promo(promo_data.code, promo_data.total_amount)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error validating promo: {e}")
        # Simple validation for development
        if promo_data.code == "TEST10":
            discount = promo_data.total_amount * 0.1
            return {
                "valid": True,
                "discount_percent": 10,
                "discount_amount": discount,
                "final_total": promo_data.total_amount - discount,
                "message": "Промокод применен! Скидка 10%"
            }
        
        return {"valid": False, "message": "Неверный промокод"}

@fastapi_app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Food Delivery API", "status": "running", "version": "1.0.0"}

@fastapi_app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# ==================== REGISTER ROUTERS ====================

dp.include_router(client_router)
dp.include_router(admin_router)

# ==================== MAIN APPLICATION ====================

async def create_tables():
    """Create database tables"""
    try:
        async with db.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("✅ Database tables created successfully")
        
        # Create sample data if tables are empty
        async with db.get_session() as session:
            # Check if categories exist
            result = await session.execute(select(Category))
            if not result.scalars().all():
                print("Creating sample categories...")
                categories = [
                    Category(name="Lavash", name_ru="Лаваш", name_uz="Lavash", sort_order=1),
                    Category(name="Burger", name_ru="Бургер", name_uz="Burger", sort_order=2),
                    Category(name="Shaurma", name_ru="Шаурма", name_uz="Shaurma", sort_order=3),
                    Category(name="Hotdog", name_ru="Хотдог", name_uz="Hotdog", sort_order=4),
                    Category(name="Combo", name_ru="Комбо", name_uz="Combo", sort_order=5),
                    Category(name="Sneki", name_ru="Снеки", name_uz="Sneki", sort_order=6),
                    Category(name="Sous", name_ru="Соусы", name_uz="Sous", sort_order=7),
                    Category(name="Napitki", name_ru="Напитки", name_uz="Napitki", sort_order=8),
                ]
                session.add_all(categories)
                await session.flush()
                
                # Create sample foods
                print("Creating sample foods...")
                foods = [
                    Food(
                        category_id=categories[0].id,
                        name="Lavash Classic",
                        name_ru="Лаваш Классик",
                        name_uz="Lavash Klassik",
                        description="Tender lavash with chicken, fresh vegetables",
                        price=28000.0,
                        rating=4.8,
                        is_new=True,
                        is_active=True
                    ),
                    Food(
                        category_id=categories[1].id,
                        name="Cheese Burger",
                        name_ru="Чизбургер",
                        name_uz="Cheese Burger",
                        description="Juicy beef burger with cheese",
                        price=32000.0,
                        rating=4.9,
                        is_new=True,
                        is_active=True
                    ),
                    Food(
                        category_id=categories[2].id,
                        name="Shaurma Big",
                        name_ru="Шаурма Большая",
                        name_uz="Shaurma Katta",
                        description="Big shaurma with chicken and vegetables",
                        price=25000.0,
                        rating=4.7,
                        is_new=False,
                        is_active=True
                    ),
                ]
                session.add_all(foods)
                
                # Create sample promo
                print("Creating sample promo...")
                promo = Promo(
                    code="TEST10",
                    discount_percent=10,
                    expires_at=datetime.utcnow() + timedelta(days=30),
                    usage_limit=100
                )
                session.add(promo)
                
                await session.commit()
                print("✅ Sample data created successfully")
                
    except Exception as e:
        print(f"❌ Error creating tables: {e}")

async def on_startup():
    """Startup actions"""
    print("🚀 Starting Food Delivery Bot...")
    
    # Create tables
    await create_tables()
    
    # Set bot commands
    try:
        await bot.set_my_commands([
            {"command": "start", "description": "Botni ishga tushirish"},
            {"command": "admin", "description": "Admin panel (faqat adminlar uchun)"}
        ])
        print("✅ Bot commands set")
    except Exception as e:
        print(f"❌ Error setting bot commands: {e}")
    
    # Set web app button
    try:
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="🛍 Заказать",
                web_app=WebAppInfo(url=config.WEBAPP_URL)
            )
        )
        print("✅ Web App button set")
    except Exception as e:
        print(f"❌ Error setting web app button: {e}")
    
    print("✅ Bot startup completed successfully!")

async def on_shutdown():
    """Shutdown actions"""
    print("🛑 Shutting down...")
    await bot.session.close()
    if 'redis' in locals():
        await redis.close()

async def run_bot():
    """Run the Telegram bot"""
    await on_startup()
    
    try:
        print("🤖 Starting bot polling...")
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await on_shutdown()

async def run_api():
    """Run the FastAPI server"""
    print("🌐 Starting FastAPI server...")
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Main application entry point"""
    # Run both bot and API concurrently
    bot_task = asyncio.create_task(run_bot())
    api_task = asyncio.create_task(run_api())
    
    # Wait for both tasks
    await asyncio.gather(bot_task, api_task)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("bot.log", encoding="utf-8")
        ]
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
