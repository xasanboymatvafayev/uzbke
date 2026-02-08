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
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager
from decimal import Decimal

import asyncpg
from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, relationship
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text,
    BigInteger, JSON, func, select, update, delete, and_, or_, Numeric, Table
)
from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardMarkup, KeyboardButton, WebAppInfo, ReplyKeyboardRemove,
    MenuButtonWebApp, WebAppData, ContentType, LabeledPrice, PreCheckoutQuery
)
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import aiohttp
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
import uvicorn
from pydantic_settings import BaseSettings

# ==================== CONFIGURATION ====================

class Settings(BaseSettings):
    BOT_TOKEN: str = "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4"
    ADMIN_IDS: List[int] = [6365371142]
    DB_URL: str = "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway"
    REDIS_URL: str = "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411"
    SHOP_CHANNEL_ID: int = -1003530497437
    COURIER_CHANNEL_ID: int = -1003707946746
    WEBAPP_URL: str = "https://mainsufooduz.netlify.app"
    API_URL: str = "https://uzbke-production.up.railway.app"
    BOT_USERNAME: str = "mainsu_food_bot"
    SECRET_KEY: str = "mainsu_food_secret_key_2024"
    WEBHOOK_URL: Optional[str] = None
    WEBHOOK_PATH: str = "/webhook"
    
    class Config:
        env_file = ".env"

settings = Settings()

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
    referrals = relationship("User", foreign_keys=[ref_by_user_id], back_populates="referrer")
    referrer = relationship("User", foreign_keys=[ref_by_user_id], back_populates="referrals", remote_side=[id])

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
    courier = relationship("Courier")
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
    
    orders = relationship("Order", back_populates="courier")

class ReferralStat(Base):
    __tablename__ = "referral_stats"
    
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    ref_count: Mapped[int] = mapped_column(Integer, default=0)
    orders_count: Mapped[int] = mapped_column(Integer, default=0)
    delivered_count: Mapped[int] = mapped_column(Integer, default=0)
    last_promo_given: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    user = relationship("User", back_populates="ref_stats")

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

db = Database(settings.DB_URL)

# ==================== REDIS STORAGE ====================

redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
storage = RedisStorage(redis=redis)

# ==================== BOT INITIALIZATION ====================

bot = Bot(token=settings.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
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
        # Parse initData
        data_pairs = init_data.split('&')
        hash_str = None
        data_check_string_parts = []
        
        for pair in data_pairs:
            key, value = pair.split('=')
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
            msg=settings.BOT_TOKEN.encode(),
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
        logging.error(f"Error verifying initdata: {e}")
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
                await session.execute(
                    update(ReferralStat)
                    .where(ReferralStat.user_id == ref_id)
                    .values(ref_count=ReferralStat.ref_count + 1)
                )
                
                # Check if referrer exists in stats
                result = await session.execute(
                    select(ReferralStat).where(ReferralStat.user_id == ref_id)
                )
                if not result.scalar_one_or_none():
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
    
    @staticmethod
    async def update_user_phone(tg_id: int, phone: str):
        async with db.get_session() as session:
            await session.execute(
                update(User)
                .where(User.tg_id == tg_id)
                .values(phone=phone)
            )

class FoodService:
    @staticmethod
    async def get_all_foods(active_only: bool = True) -> List[Dict]:
        async with db.get_session() as session:
            query = select(Food, Category.name).join(Category)
            if active_only:
                query = query.where(Food.is_active == True)
            
            result = await session.execute(query)
            foods = []
            for food, category_name in result.all():
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
                    "category_name": category_name
                })
            return foods
    
    @staticmethod
    async def get_foods_by_category(category_id: int, active_only: bool = True) -> List[Dict]:
        async with db.get_session() as session:
            query = select(Food).where(Food.category_id == category_id)
            if active_only:
                query = query.where(Food.is_active == True)
            
            result = await session.execute(query)
            return [food for food in result.scalars().all()]
    
    @staticmethod
    async def get_categories(active_only: bool = True) -> List[Category]:
        async with db.get_session() as session:
            query = select(Category)
            if active_only:
                query = query.where(Category.is_active == True)
            query = query.order_by(Category.sort_order)
            
            result = await session.execute(query)
            return result.scalars().all()

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
                    await session.execute(
                        update(Promo)
                        .where(Promo.id == promo.id)
                        .values(used_count=Promo.used_count + 1)
                    )
            
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
                    food_id=item["food_id"],
                    name_snapshot=item["name"],
                    price_snapshot=item["price"],
                    qty=item["qty"],
                    line_total=item["price"] * item["qty"]
                )
                session.add(order_item)
            
            # Update user stats
            user_stats = await session.execute(
                select(ReferralStat).where(ReferralStat.user_id == user_id)
            )
            stats = user_stats.scalar_one_or_none()
            if stats:
                await session.execute(
                    update(ReferralStat)
                    .where(ReferralStat.user_id == user_id)
                    .values(orders_count=ReferralStat.orders_count + 1)
                )
            
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
    
    @staticmethod
    async def update_order_status(order_id: int, status: OrderStatus, courier_id: Optional[int] = None):
        async with db.get_session() as session:
            update_data = {"status": status.value, "updated_at": datetime.utcnow()}
            if courier_id:
                update_data["courier_id"] = courier_id
            if status == OrderStatus.DELIVERED:
                update_data["delivered_at"] = datetime.utcnow()
                
                # Update user delivered count
                order = await session.get(Order, order_id)
                if order:
                    await session.execute(
                        update(ReferralStat)
                        .where(ReferralStat.user_id == order.user_id)
                        .values(delivered_count=ReferralStat.delivered_count + 1)
                    )
            
            await session.execute(
                update(Order)
                .where(Order.id == order_id)
                .values(**update_data)
            )
    
    @staticmethod
    async def get_active_orders() -> List[Order]:
        async with db.get_session() as session:
            result = await session.execute(
                select(Order)
                .where(
                    and_(
                        Order.status.in_([
                            OrderStatus.NEW.value,
                            OrderStatus.CONFIRMED.value,
                            OrderStatus.COOKING.value,
                            OrderStatus.COURIER_ASSIGNED.value,
                            OrderStatus.OUT_FOR_DELIVERY.value
                        ])
                    )
                )
                .order_by(Order.created_at.desc())
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
    
    @staticmethod
    async def create_promo(
        code: str,
        discount_percent: int,
        expires_at: Optional[datetime] = None,
        usage_limit: Optional[int] = None,
        created_by: Optional[int] = None
    ) -> Promo:
        async with db.get_session() as session:
            promo = Promo(
                code=code,
                discount_percent=discount_percent,
                expires_at=expires_at,
                usage_limit=usage_limit,
                created_by=created_by
            )
            session.add(promo)
            await session.commit()
            return promo

class CourierService:
    @staticmethod
    async def get_active_couriers() -> List[Courier]:
        async with db.get_session() as session:
            result = await session.execute(
                select(Courier)
                .where(Courier.is_active == True)
                .order_by(Courier.name)
            )
            return result.scalars().all()
    
    @staticmethod
    async def get_courier_by_chat_id(chat_id: int) -> Optional[Courier]:
        async with db.get_session() as session:
            result = await session.execute(
                select(Courier).where(Courier.chat_id == chat_id)
            )
            return result.scalar_one_or_none()
    
    @staticmethod
    async def assign_order_to_courier(order_id: int, courier_id: int):
        async with db.get_session() as session:
            await session.execute(
                update(Order)
                .where(Order.id == order_id)
                .values(
                    courier_id=courier_id,
                    status=OrderStatus.COURIER_ASSIGNED.value,
                    updated_at=datetime.utcnow()
                )
            )

class StatsService:
    @staticmethod
    async def get_daily_stats(date: datetime = None):
        if date is None:
            date = datetime.now()
        
        async with db.get_session() as session:
            # Orders count
            result = await session.execute(
                select(func.count(Order.id))
                .where(func.date(Order.created_at) == date.date())
            )
            orders_count = result.scalar() or 0
            
            # Delivered count
            result = await session.execute(
                select(func.count(Order.id))
                .where(
                    and_(
                        func.date(Order.created_at) == date.date(),
                        Order.status == OrderStatus.DELIVERED.value
                    )
                )
            )
            delivered_count = result.scalar() or 0
            
            # Revenue
            result = await session.execute(
                select(func.sum(Order.final_total))
                .where(
                    and_(
                        func.date(Order.created_at) == date.date(),
                        Order.status == OrderStatus.DELIVERED.value
                    )
                )
            )
            revenue = result.scalar() or 0.0
            
            # Top foods
            result = await session.execute(
                select(
                    OrderItem.name_snapshot,
                    func.sum(OrderItem.qty).label("total_qty")
                )
                .join(Order)
                .where(func.date(Order.created_at) == date.date())
                .group_by(OrderItem.name_snapshot)
                .order_by(func.sum(OrderItem.qty).desc())
                .limit(5)
            )
            top_foods = result.all()
            
            return {
                "orders_count": orders_count,
                "delivered_count": delivered_count,
                "revenue": revenue,
                "top_foods": [{"name": name, "quantity": qty} for name, qty in top_foods]
            }
    
    @staticmethod
    async def get_user_stats(user_id: int) -> Dict:
        async with db.get_session() as session:
            result = await session.execute(
                select(ReferralStat)
                .where(ReferralStat.user_id == user_id)
            )
            stats = result.scalar_one_or_none()
            
            if stats:
                return {
                    "ref_count": stats.ref_count,
                    "orders_count": stats.orders_count,
                    "delivered_count": stats.delivered_count,
                    "last_promo_given": stats.last_promo_given
                }
            return {"ref_count": 0, "orders_count": 0, "delivered_count": 0, "last_promo_given": None}

# ==================== TELEGRAM NOTIFICATION SERVICE ====================

class TelegramNotifyService:
    @staticmethod
    async def send_order_notification(order: Order):
        """Send order notification to admin channel"""
        try:
            # Format order items
            items_text = ""
            for item in order.items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {item.price_snapshot * item.qty:,} сум\n"
            
            # Create message text
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
                chat_id=settings.SHOP_CHANNEL_ID,
                text=message_text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML
            )
            
            # Save message ID to order
            async with db.get_session() as session:
                await session.execute(
                    update(Order)
                    .where(Order.id == order.id)
                    .values(channel_message_id=message.message_id)
                )
            
        except Exception as e:
            logging.error(f"Error sending order notification: {e}")
    
    @staticmethod
    async def update_order_message(order: Order):
        """Update order message in admin channel"""
        try:
            if not order.channel_message_id:
                return
            
            # Format order items
            items_text = ""
            for item in order.items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {item.price_snapshot * item.qty:,} сум\n"
            
            # Status emoji
            status_emoji = {
                OrderStatus.NEW.value: "🆕",
                OrderStatus.CONFIRMED.value: "✅",
                OrderStatus.COOKING.value: "🍳",
                OrderStatus.COURIER_ASSIGNED.value: "🚴",
                OrderStatus.OUT_FOR_DELIVERY.value: "📦",
                OrderStatus.DELIVERED.value: "🎉",
                OrderStatus.CANCELED.value: "❌"
            }.get(order.status, "📝")
            
            # Create message text
            message_text = (
                f"{status_emoji} <b>Заказ №{order.order_number}</b> - {order.status}\n"
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
            
            # Update keyboard based on status
            keyboard = None
            if order.status in [OrderStatus.NEW.value, OrderStatus.CONFIRMED.value, OrderStatus.COOKING.value]:
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
            elif order.status == OrderStatus.COURIER_ASSIGNED.value:
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(text="✅ Принято курьером", callback_data=f"courier_accepted:{order.id}")
                    ],
                    [
                        InlineKeyboardButton(text="📦 Доставлен", callback_data=f"order_delivered:{order.id}")
                    ]
                ])
            
            # Edit message
            await bot.edit_message_text(
                chat_id=settings.SHOP_CHANNEL_ID,
                message_id=order.channel_message_id,
                text=message_text,
                reply_markup=keyboard,
                parse_mode=ParseMode.HTML
            )
            
        except Exception as e:
            logging.error(f"Error updating order message: {e}")
    
    @staticmethod
    async def send_to_courier(order: Order, courier: Courier):
        """Send order to courier"""
        try:
            # Format order items
            items_text = ""
            for item in order.items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {item.price_snapshot * item.qty:,} сум\n"
            
            # Create message text
            message_text = (
                f"🚴 <b>Новый заказ для доставки</b>\n"
                f"🆔 <b>Номер:</b> {order.order_number}\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {order.final_total:,.0f} сум\n"
                f"📍 <b>Локация:</b> <a href='https://maps.google.com/?q={order.location_lat},{order.location_lng}'>Показать на карте</a>\n"
                f"📝 <b>Комментарий:</b> {order.comment or 'Нет'}\n\n"
                f"🍽️ <b>Заказ:</b>\n{items_text}"
            )
            
            # Create inline keyboard
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Принял заказ", callback_data=f"courier_accept:{order.id}"),
                    InlineKeyboardButton(text="📦 Доставил", callback_data=f"courier_delivered:{order.id}")
                ]
            ])
            
            # Send to courier
            if settings.COURIER_CHANNEL_ID:
                await bot.send_message(
                    chat_id=settings.COURIER_CHANNEL_ID,
                    text=message_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML
                )
            else:
                # Send to individual courier
                await bot.send_message(
                    chat_id=courier.chat_id,
                    text=message_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML
                )
            
        except Exception as e:
            logging.error(f"Error sending to courier: {e}")
    
    @staticmethod
    async def notify_user(order: Order, message: str):
        """Send notification to user about order status"""
        try:
            user = await UserService.get_user_by_tg_id(order.user_id)
            if user:
                await bot.send_message(
                    chat_id=user.tg_id,
                    text=message,
                    parse_mode=ParseMode.HTML
                )
        except Exception as e:
            logging.error(f"Error notifying user: {e}")

# ==================== FASTAPI ROUTES ====================

@fastapi_app.get("/api/foods")
async def get_foods(request: Request):
    """Get all active foods"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        raise HTTPException(status_code=403, detail="Invalid initData")
    
    foods = await FoodService.get_all_foods(active_only=True)
    return JSONResponse(content=foods)

@fastapi_app.get("/api/categories")
async def get_categories(request: Request):
    """Get all active categories"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        raise HTTPException(status_code=403, detail="Invalid initData")
    
    categories = await FoodService.get_categories(active_only=True)
    result = []
    for cat in categories:
        foods = await FoodService.get_foods_by_category(cat.id, active_only=True)
        result.append({
            "id": cat.id,
            "name": cat.name,
            "name_ru": cat.name_ru,
            "name_uz": cat.name_uz,
            "is_active": cat.is_active,
            "image_url": cat.image_url,
            "foods_count": len(foods)
        })
    return JSONResponse(content=result)

@fastapi_app.post("/api/promo/validate")
async def validate_promo(promo_data: PromoValidate, request: Request):
    """Validate promo code"""
    # Check Telegram initData
    init_data = request.headers.get("X-Telegram-Init-Data")
    if not init_data or not verify_telegram_initdata(init_data):
        raise HTTPException(status_code=403, detail="Invalid initData")
    
    result = await PromoService.validate_promo(promo_data.code, promo_data.total_amount)
    return JSONResponse(content=result)

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
                KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=settings.WEBAPP_URL)),
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
                [InlineKeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=settings.WEBAPP_URL))]
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
    
    stats = await StatsService.get_user_stats(user.id)
    
    # Check if user deserves a promo code
    if stats["ref_count"] >= 3 and (not stats["last_promo_given"] or 
                                    (datetime.utcnow() - stats["last_promo_given"]).days > 30):
        # Generate promo code
        promo_code = f"REF{user.tg_id}{datetime.now().strftime('%m%d')}"
        await PromoService.create_promo(
            code=promo_code,
            discount_percent=15,
            expires_at=datetime.utcnow() + timedelta(days=30),
            usage_limit=1,
            created_by=user.id
        )
        
        # Update last promo given
        async with db.get_session() as session:
            await session.execute(
                update(ReferralStat)
                .where(ReferralStat.user_id == user.id)
                .values(last_promo_given=datetime.utcnow())
            )
        
        promo_message = f"\n\n🎉 Поздравляем! Вы получили промо-код: {promo_code} (скидка 15%)"
    else:
        promo_message = "\n\n🎁 Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
    
    invite_text = (
        f"За приглашение друга, вы можете получить промо-код от нас\n"
        f"👥 Вы пригласили {stats['ref_count']} человек\n"
        f"🛒 Оформили заказов: {stats['orders_count']}\n"
        f"💰 Оплатили заказов: {stats['delivered_count']}\n"
        f"👤 Ваша реферальная ссылка: https://t.me/{settings.BOT_USERNAME}?start={user.tg_id}"
        f"{promo_message}"
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
                await message.answer("Пожалуйста, начните с команды /start")
                return
            
            # Validate total
            if data["total"] < 50000:
                await message.answer("Минимальная сумма заказа 50,000 сум")
                return
            
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
            await TelegramNotifyService.send_order_notification(order)
            
    except Exception as e:
        logging.error(f"Error handling web app data: {e}")
        await message.answer("Произошла ошибка при обработке заказа. Пожалуйста, попробуйте позже.")

# Admin router
admin_router = Router()

def is_admin(user_id: int) -> bool:
    """Check if user is admin"""
    return user_id in settings.ADMIN_IDS

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
    
    # Get daily stats
    stats = await StatsService.get_daily_stats()
    
    # Format top foods
    top_foods_text = ""
    for i, food in enumerate(stats["top_foods"], 1):
        top_foods_text += f"{i}. {food['name']} - {food['quantity']} шт\n"
    
    stats_text = (
        f"📊 <b>Статистика на сегодня</b>\n\n"
        f"📦 Заказов: {stats['orders_count']}\n"
        f"🎉 Доставлено: {stats['delivered_count']}\n"
        f"💰 Выручка: {stats['revenue']:,.0f} сум\n\n"
        f"🏆 <b>Топ товаров:</b>\n{top_foods_text}"
    )
    
    await callback.message.edit_text(stats_text, parse_mode=ParseMode.HTML)

@admin_router.callback_query(F.data == "admin_active_orders")
async def admin_active_orders(callback: CallbackQuery):
    """Show active orders"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    orders = await OrderService.get_active_orders()
    
    if not orders:
        await callback.message.edit_text("📭 Фаол буюртмалар йўқ")
        return
    
    # Create keyboard with orders
    keyboard_buttons = []
    for order in orders:
        status_emoji = {
            OrderStatus.NEW.value: "🆕",
            OrderStatus.CONFIRMED.value: "✅",
            OrderStatus.COOKING.value: "🍳",
            OrderStatus.COURIER_ASSIGNED.value: "🚴",
            OrderStatus.OUT_FOR_DELIVERY.value: "📦"
        }.get(order.status, "📝")
        
        button_text = f"{status_emoji} №{order.order_number} - {order.final_total:,.0f} сум"
        keyboard_buttons.append([InlineKeyboardButton(
            text=button_text,
            callback_data=f"view_order:{order.id}"
        )])
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons + [
        [InlineKeyboardButton(text="🔙 Ортга", callback_data="back_to_admin")]
    ])
    
    await callback.message.edit_text(
        f"📦 Фаол буюртмалар ({len(orders)} та)",
        reply_markup=keyboard
    )

@admin_router.callback_query(F.data.startswith("view_order:"))
async def view_order(callback: CallbackQuery):
    """View order details"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with db.get_session() as session:
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if not order:
            await callback.answer("Буюртма топилмади")
            return
        
        # Format order items
        items_text = ""
        for item in order.items:
            items_text += f"• {item.name_snapshot} x{item.qty} = {item.line_total:,} сум\n"
        
        order_text = (
            f"🆔 <b>Буюртма №{order.order_number}</b>\n"
            f"👤 <b>Мижоз:</b> {order.customer_name}\n"
            f"📞 <b>Телефон:</b> {order.phone}\n"
            f"💰 <b>Сумма:</b> {order.total:,.0f} сум\n"
            f"🎁 <b>Чегирма:</b> {order.discount_amount:,.0f} сум\n"
            f"💵 <b>Жами:</b> {order.final_total:,.0f} сум\n"
            f"📦 <b>Статус:</b> {order.status}\n"
            f"🕒 <b>Вақт:</b> {order.created_at.strftime('%H:%M %d.%m.%Y')}\n"
            f"📍 <b>Локация:</b> <a href='https://maps.google.com/?q={order.location_lat},{order.location_lng}'>Харитада кўриш</a>\n"
            f"📝 <b>Изоҳ:</b> {order.comment or 'Йўқ'}\n\n"
            f"🍽️ <b>Буюртма:</b>\n{items_text}"
        )
        
        # Create action buttons
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Тасдиқлаш", callback_data=f"confirm_order:{order.id}"),
                InlineKeyboardButton(text="🍳 Тайёрланиш", callback_data=f"cooking_order:{order.id}")
            ],
            [
                InlineKeyboardButton(text="🚴 Курьер", callback_data=f"assign_courier:{order.id}")
            ],
            [
                InlineKeyboardButton(text="❌ Бекор қилиш", callback_data=f"cancel_order:{order.id}")
            ],
            [
                InlineKeyboardButton(text="🔙 Ортга", callback_data="admin_active_orders")
            ]
        ])
        
        await callback.message.edit_text(order_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

# Order status handlers
@admin_router.callback_query(F.data.startswith("confirm_order:"))
async def confirm_order(callback: CallbackQuery):
    """Confirm order"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Update order status
    await OrderService.update_order_status(order_id, OrderStatus.CONFIRMED)
    
    # Get order for notification
    async with db.get_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if order:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"✅ Ваш заказ №{order.order_number} подтвержден и готовится!"
            )
    
    await callback.answer("Буюртма тасдиқланди")
    await callback.message.edit_reply_markup(reply_markup=None)

@admin_router.callback_query(F.data.startswith("cooking_order:"))
async def cooking_order(callback: CallbackQuery):
    """Set order to cooking"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Update order status
    await OrderService.update_order_status(order_id, OrderStatus.COOKING)
    
    # Get order for notification
    async with db.get_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if order:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"🍳 Ваш заказ №{order.order_number} готовится!"
            )
    
    await callback.answer("Буюртма тайёрланиш жараёнида")
    await callback.message.edit_reply_markup(reply_markup=None)

@admin_router.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier(callback: CallbackQuery):
    """Assign courier to order"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Get active couriers
    couriers = await CourierService.get_active_couriers()
    
    if not couriers:
        await callback.answer("Фаол курьерлар йўқ")
        return
    
    # Create keyboard with couriers
    keyboard_buttons = []
    for courier in couriers:
        keyboard_buttons.append([InlineKeyboardButton(
            text=f"🚴 {courier.name}",
            callback_data=f"assign_to_courier:{order_id}:{courier.id}"
        )])
    
    keyboard_buttons.append([InlineKeyboardButton(
        text="🔙 Ортга",
        callback_data=f"view_order:{order_id}"
    )])
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
    
    await callback.message.edit_text(
        f"🚴 Курьер танланг (Буюртма №{order_id})",
        reply_markup=keyboard
    )

@admin_router.callback_query(F.data.startswith("assign_to_courier:"))
async def assign_to_courier(callback: CallbackQuery):
    """Assign order to specific courier"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    # Assign courier and update status
    await CourierService.assign_order_to_courier(order_id, courier_id)
    await OrderService.update_order_status(order_id, OrderStatus.COURIER_ASSIGNED)
    
    # Get order and courier
    async with db.get_session() as session:
        # Get order
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        # Get courier
        result = await session.execute(select(Courier).where(Courier.id == courier_id))
        courier = result.scalar_one_or_none()
        
        if order and courier:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Send to courier
            await TelegramNotifyService.send_to_courier(order, courier)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"🚴 Ваш заказ №{order.order_number} передан курьеру {courier.name}!"
            )
    
    await callback.answer(f"Курьерга топширилди")
    await callback.message.edit_reply_markup(reply_markup=None)

@admin_router.callback_query(F.data.startswith("cancel_order:"))
async def cancel_order(callback: CallbackQuery):
    """Cancel order"""
    if not is_admin(callback.from_user.id):
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    # Update order status
    await OrderService.update_order_status(order_id, OrderStatus.CANCELED)
    
    # Get order for notification
    async with db.get_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if order:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"❌ Ваш заказ №{order.order_number} отменен"
            )
    
    await callback.answer("Буюртма бекор қилинди")
    await callback.message.edit_reply_markup(reply_markup=None)

# Courier handlers
courier_router = Router()

@courier_router.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept(callback: CallbackQuery):
    """Courier accepts order"""
    order_id = int(callback.data.split(":")[1])
    
    # Check if user is a courier
    courier = await CourierService.get_courier_by_chat_id(callback.from_user.id)
    if not courier:
        await callback.answer("Сиз курьер эмассиз")
        return
    
    # Update order status
    await OrderService.update_order_status(order_id, OrderStatus.OUT_FOR_DELIVERY, courier.id)
    
    # Get order for notification
    async with db.get_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if order and order.courier_id == courier.id:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"📦 Ваш заказ №{order.order_number} принят курьером и в пути!"
            )
    
    await callback.answer("Буюртма қабул қилинди")
    await callback.message.edit_reply_markup(reply_markup=None)

@courier_router.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered(callback: CallbackQuery):
    """Courier marks order as delivered"""
    order_id = int(callback.data.split(":")[1])
    
    # Check if user is a courier
    courier = await CourierService.get_courier_by_chat_id(callback.from_user.id)
    if not courier:
        await callback.answer("Сиз курьер эмассиз")
        return
    
    # Update order status
    await OrderService.update_order_status(order_id, OrderStatus.DELIVERED, courier.id)
    
    # Get order for notification
    async with db.get_session() as session:
        result = await session.execute(select(Order).where(Order.id == order_id))
        order = result.scalar_one_or_none()
        
        if order and order.courier_id == courier.id:
            # Update channel message
            await TelegramNotifyService.update_order_message(order)
            
            # Notify user
            await TelegramNotifyService.notify_user(
                order,
                f"🎉 Ваш заказ №{order.order_number} успешно доставлен! Спасибо за заказ!"
            )
    
    await callback.answer("Буюртма етказилди")
    await callback.message.edit_reply_markup(reply_markup=None)

# Channel post handlers
@dp.channel_post()
async def handle_channel_post(message: Message):
    """Handle channel posts"""
    pass

# ==================== REGISTER ROUTERS ====================

dp.include_router(client_router)
dp.include_router(admin_router)
dp.include_router(courier_router)

# ==================== MAIN APPLICATION ====================

async def create_tables():
    """Create database tables"""
    async with db.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create sample data if needed
    async with db.get_session() as session:
        # Check if categories exist
        result = await session.execute(select(Category))
        if not result.scalars().all():
            # Create sample categories
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
            await session.commit()

async def on_startup():
    """Startup actions"""
    logging.info("Starting up...")
    
    # Create tables
    await create_tables()
    
    # Set bot commands
    await bot.set_my_commands([
        {"command": "start", "description": "Botni ishga tushirish"},
        {"command": "admin", "description": "Admin panel (faqat adminlar uchun)"}
    ])
    
    # Set web app button
    await bot.set_chat_menu_button(
        menu_button=MenuButtonWebApp(
            text="🛍 Заказать",
            web_app=WebAppInfo(url=settings.WEBAPP_URL)
        )
    )
    
    logging.info("Bot started!")

async def on_shutdown():
    """Shutdown actions"""
    logging.info("Shutting down...")
    await bot.session.close()
    await redis.close()

async def main():
    """Main application entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Startup actions
    await on_startup()
    
    # Start polling
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await on_shutdown()

if __name__ == "__main__":
    # Run both FastAPI and bot
    import threading
    
    # Start FastAPI in a separate thread
    def run_fastapi():
        uvicorn.run(
            fastapi_app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    
    # Start FastAPI thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Run bot in main thread
    asyncio.run(main())
