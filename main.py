#!/usr/bin/env python3
"""
Food Delivery Telegram Bot
Production-ready system with WebApp
"""

import asyncio
import logging
from typing import List, Optional
from datetime import datetime, timezone
import json
import secrets
import hmac
import hashlib
import urllib.parse

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import (
    Message, CallbackQuery, WebAppInfo, 
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ReplyKeyboardRemove, WebAppData
)
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
import redis.asyncio as redis

from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
)
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, BigInteger, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

# ============================================================================
# Configuration
# ============================================================================

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    ADMIN_IDS = [int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]
    DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/food_delivery")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    SHOP_CHANNEL_ID = os.getenv("SHOP_CHANNEL_ID")
    COURIER_CHANNEL_ID = os.getenv("COURIER_CHANNEL_ID")
    WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.netlify.app/webapp")
    MIN_ORDER_AMOUNT = 50000  # 50,000 сум
    
    @staticmethod
    def get_bot_username():
        return getattr(Config, '_BOT_USERNAME', 'your_bot_username')

config = Config()

# ============================================================================
# Database Setup
# ============================================================================

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    tg_id = Column(BigInteger, unique=True, nullable=False, index=True)
    username = Column(String(100), nullable=True)
    full_name = Column(String(200), nullable=False)
    joined_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ref_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    orders = relationship("Order", back_populates="user")
    referrals = relationship("User", foreign_keys=[ref_by_user_id])
    ref_by = relationship("User", foreign_keys=[ref_by_user_id], remote_side=[id], back_populates="referrals")

class Category(Base):
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    foods = relationship("Food", back_populates="category")

class Food(Base):
    __tablename__ = 'foods'
    
    id = Column(Integer, primary_key=True)
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Integer, nullable=False)  # in sum
    rating = Column(Float, default=4.5)
    is_new = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    image_url = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    category = relationship("Category", back_populates="foods")
    order_items = relationship("OrderItem", back_populates="food")

class Order(Base):
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    customer_name = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    comment = Column(Text, nullable=True)
    total = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default='NEW', index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    location_lat = Column(Float, nullable=False)
    location_lng = Column(Float, nullable=False)
    courier_id = Column(Integer, ForeignKey('couriers.id'), nullable=True)
    promo_id = Column(Integer, ForeignKey('promos.id'), nullable=True)
    
    user = relationship("User", back_populates="orders")
    courier = relationship("Courier", back_populates="orders")
    promo = relationship("Promo")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    food_id = Column(Integer, ForeignKey('foods.id'), nullable=True)
    name_snapshot = Column(String(200), nullable=False)
    price_snapshot = Column(Integer, nullable=False)
    qty = Column(Integer, nullable=False, default=1)
    line_total = Column(Integer, nullable=False)
    
    order = relationship("Order", back_populates="items")
    food = relationship("Food", back_populates="order_items")

class Promo(Base):
    __tablename__ = 'promos'
    
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False, index=True)
    discount_percent = Column(Integer, nullable=False)  # 1-90
    expires_at = Column(DateTime(timezone=True), nullable=True)
    usage_limit = Column(Integer, nullable=True)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    orders = relationship("Order", back_populates="promo")

class Courier(Base):
    __tablename__ = 'couriers'
    
    id = Column(Integer, primary_key=True)
    chat_id = Column(BigInteger, unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    orders = relationship("Order", back_populates="courier")

# ============================================================================
# Database Engine & Session
# ============================================================================

engine = create_async_engine(config.DB_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_db_session() -> AsyncSession:
    """Create database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logging.info("Database initialized")

# ============================================================================
# Redis Storage for FSM
# ============================================================================

redis_client = redis.from_url(config.REDIS_URL)
storage = RedisStorage(redis=redis_client)

# ============================================================================
# Bot & Dispatcher
# ============================================================================

bot = Bot(token=config.BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher(storage=storage)

# ============================================================================
# Helper Functions
# ============================================================================

async def set_bot_username():
    """Set bot username from Telegram API"""
    try:
        me = await bot.get_me()
        config._BOT_USERNAME = me.username
        logging.info(f"Bot username set: @{me.username}")
    except Exception as e:
        logging.error(f"Failed to get bot info: {e}")
        config._BOT_USERNAME = "food_delivery_bot"

def verify_telegram_init_data(init_data: str) -> bool:
    """Verify Telegram WebApp initData signature"""
    try:
        # Parse init data
        parsed = urllib.parse.parse_qs(init_data)
        
        # Get hash from init data
        hash_str = parsed.get('hash', [''])[0]
        if not hash_str:
            return False
        
        # Remove hash from data
        data_pairs = []
        for key, values in parsed.items():
            if key != 'hash':
                data_pairs.append(f"{key}={values[0]}")
        
        # Sort alphabetically
        data_pairs.sort()
        data_check_string = "\n".join(data_pairs)
        
        # Calculate secret
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
        logging.error(f"Error verifying init data: {e}")
        return False

# ============================================================================
# Bot Handlers - Client
# ============================================================================

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Handle /start command with referral"""
    args = message.text.split()
    ref_id = None
    
    # Check for referral
    if len(args) > 1:
        try:
            ref_id = int(args[1])
        except ValueError:
            pass
    
    async with AsyncSessionLocal() as session:
        # Check if user exists
        result = await session.execute(
            select(User).where(User.tg_id == message.from_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            ref_by_user = None
            if ref_id:
                # Check if referrer exists and not self-referral
                if ref_id != message.from_user.id:
                    result = await session.execute(
                        select(User).where(User.tg_id == ref_id)
                    )
                    ref_by_user = result.scalar_one_or_none()
            
            user = User(
                tg_id=message.from_user.id,
                username=message.from_user.username,
                full_name=message.from_user.full_name,
                ref_by_user_id=ref_by_user.id if ref_by_user else None
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        
        # Send welcome message
        keyboard = ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=config.WEBAPP_URL))
                ],
                [
                    KeyboardButton(text="📦 Мои заказы"),
                    KeyboardButton(text="ℹ️ Информация о нас")
                ],
                [
                    KeyboardButton(text="👥 Пригласить друга")
                ]
            ],
            resize_keyboard=True,
            input_field_placeholder="Выберите действие..."
        )
        
        await message.answer(
            f"Добро пожаловать в FIESTA! {message.from_user.full_name}\n\n"
            "Для заказа перейдите по кнопке ➡️\n"
            "🛍 Заказать",
            reply_markup=keyboard
        )

@dp.message(lambda message: message.text == "📦 Мои заказы")
async def my_orders(message: Message):
    """Show user's orders"""
    async with AsyncSessionLocal() as session:
        # Get user
        result = await session.execute(
            select(User).where(User.tg_id == message.from_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            await message.answer("Сначала начните с /start")
            return
        
        # Get orders
        result = await session.execute(
            select(Order)
            .where(Order.user_id == user.id)
            .order_by(Order.created_at.desc())
            .limit(10)
        )
        orders = result.scalars().all()
        
        if not orders:
            await message.answer(
                "В данный момент у вас нет активных заказов в нашем магазине.\n"
                "Чтобы открыть магазин, введите команду — /shop",
                reply_markup=ReplyKeyboardRemove()
            )
            return
        
        for order in orders:
            # Get order items
            result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = result.scalars().all()
            
            items_text = "\n".join([
                f"• {item.name_snapshot} x{item.qty} = {item.line_total:,} сум"
                for item in items
            ])
            
            status_text = {
                'NEW': 'Принят',
                'CONFIRMED': 'Подтвержден',
                'COOKING': 'Готовится',
                'COURIER_ASSIGNED': 'Курьер назначен',
                'OUT_FOR_DELIVERY': 'Передан курьеру',
                'DELIVERED': 'Доставлен',
                'CANCELED': 'Отменен'
            }.get(order.status, order.status)
            
            await message.answer(
                f"🆔 Заказ №{order.order_number}\n"
                f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                f"💰 {order.total:,} сум\n"
                f"📦 {status_text}\n\n"
                f"🍽️ Заказ:\n{items_text}"
            )

@dp.message(Command("shop"))
async def cmd_shop(message: Message):
    """Shop command"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🛍 Заказать",
                    web_app=WebAppInfo(url=config.WEBAPP_URL)
                )
            ]
        ]
    )
    
    await message.answer(
        "Чтобы открыть наш магазин, нажмите кнопку ниже",
        reply_markup=keyboard
    )

@dp.message(lambda message: message.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    """About us information"""
    await message.answer(
        "🌟 Добро Пожаловать в FIESTA !\n\n"
        "📍 Наш адрес: Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢 Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙 Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)",
        disable_web_page_preview=True
    )

@dp.message(lambda message: message.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    """Referral system"""
    async with AsyncSessionLocal() as session:
        # Get user
        result = await session.execute(
            select(User).where(User.tg_id == message.from_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            await message.answer("Сначала начните с /start")
            return
        
        # Get referral stats
        # Count referrals
        result = await session.execute(
            select(func.count(User.id)).where(User.ref_by_user_id == user.id)
        )
        ref_count = result.scalar() or 0
        
        # Count user's orders
        result = await session.execute(
            select(func.count(Order.id)).where(Order.user_id == user.id)
        )
        orders_count = result.scalar() or 0
        
        # Count delivered orders
        result = await session.execute(
            select(func.count(Order.id)).where(
                Order.user_id == user.id,
                Order.status == 'DELIVERED'
            )
        )
        delivered_count = result.scalar() or 0
        
        # Check if user already has a promo for referrals
        result = await session.execute(
            select(Promo).where(
                Promo.code.like(f"REF_{user.tg_id}_%")
            )
        )
        existing_promo = result.scalar_one_or_none()
        
        # Create promo if ref_count >= 3 and no existing promo
        if ref_count >= 3 and not existing_promo:
            promo_code = f"REF_{user.tg_id}_{secrets.token_hex(3).upper()}"
            promo = Promo(
                code=promo_code,
                discount_percent=15,
                expires_at=None,  # Never expires
                usage_limit=5,
                is_active=True
            )
            session.add(promo)
            await session.commit()
            
            await message.answer(
                f"🎉 Поздравляем! Вы пригласили {ref_count} друзей.\n"
                f"Ваш промо-код на 15% скидку: {promo_code}\n"
                f"Лимит использований: 5 раз"
            )
        
        bot_username = config.get_bot_username()
        referral_link = f"https://t.me/{bot_username}?start={user.tg_id}"
        
        await message.answer(
            "За приглашение друга, вы можете получить промо-код от нас\n\n"
            f"👥 Вы пригласили {ref_count} человек\n"
            f"🛒 Оформили заказов: {orders_count}\n"
            f"💰 Оплатили заказов: {delivered_count}\n\n"
            f"👤 Ваша реферальная ссылка:\n{referral_link}\n\n"
            "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
        )

# ============================================================================
# Bot Handlers - WebApp Data
# ============================================================================

@dp.message(lambda message: message.web_app_data is not None)
async def handle_webapp_data(message: Message):
    """Handle data from WebApp"""
    try:
        data = json.loads(message.web_app_data.data)
        logging.info(f"WebApp data received: {data}")
        
        if data.get("type") == "order_create":
            await handle_order_create(message, data)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding WebApp data: {e}")
        await message.answer("Ошибка при обработке данных. Неверный формат.")
    except Exception as e:
        logging.error(f"Error processing WebApp data: {e}")
        await message.answer("Ошибка при обработке заказа. Пожалуйста, попробуйте еще раз.")

async def handle_order_create(message: Message, data: dict):
    """Create order from WebApp data"""
    # Validate total amount
    if data["total"] < config.MIN_ORDER_AMOUNT:
        await message.answer(
            f"Минимальная сумма заказа {config.MIN_ORDER_AMOUNT:,} сум. "
            f"Ваша сумма: {data['total']:,} сум"
        )
        return
    
    async with AsyncSessionLocal() as session:
        # Get or create user
        result = await session.execute(
            select(User).where(User.tg_id == message.from_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            user = User(
                tg_id=message.from_user.id,
                username=message.from_user.username,
                full_name=message.from_user.full_name
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        
        # Check promo code
        promo_id = None
        if data.get("promo_code"):
            result = await session.execute(
                select(Promo).where(
                    Promo.code == data["promo_code"],
                    Promo.is_active == True,
                    or_(
                        Promo.expires_at.is_(None),
                        Promo.expires_at > datetime.now(timezone.utc)
                    )
                )
            )
            promo = result.scalar_one_or_none()
            
            if promo and (promo.usage_limit is None or promo.used_count < promo.usage_limit):
                promo_id = promo.id
                promo.used_count += 1
        
        # Generate order number
        order_number = f"ORD-{datetime.now().strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
        
        # Create order
        order = Order(
            order_number=order_number,
            user_id=user.id,
            customer_name=data["customer_name"],
            phone=data["phone"],
            comment=data.get("comment"),
            total=data["total"],
            status="NEW",
            location_lat=data["location"]["lat"],
            location_lng=data["location"]["lng"],
            promo_id=promo_id
        )
        session.add(order)
        await session.flush()  # Get order ID without committing
        
        # Create order items
        for item in data["items"]:
            order_item = OrderItem(
                order_id=order.id,
                food_id=item.get("food_id"),
                name_snapshot=item["name"],
                price_snapshot=item["price"],
                qty=item["qty"],
                line_total=item["price"] * item["qty"]
            )
            session.add(order_item)
        
        await session.commit()
        await session.refresh(order)
        
        # Send confirmation to user
        await message.answer(
            f"✅ Ваш заказ принят!\n\n"
            f"🆔 Заказ №{order.order_number}\n"
            f"💰 Сумма: {order.total:,} сум\n"
            f"📦 Статус: Принят\n\n"
            f"Мы скоро свяжемся с вами для подтверждения."
        )
        
        # Send to admin channel
        if config.SHOP_CHANNEL_ID:
            await send_order_to_admin_channel(order, user)

async def send_order_to_admin_channel(order: Order, user: User):
    """Send order notification to admin channel"""
    async with AsyncSessionLocal() as session:
        # Get order items
        result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = "\n".join([
            f"{item.name_snapshot} x{item.qty} = {item.line_total:,} сум"
            for item in items
        ])
        
        # Create inline keyboard for admin actions
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Подтвержден",
                        callback_data=f"confirm_order:{order.id}"
                    ),
                    InlineKeyboardButton(
                        text="🍳 Готовится",
                        callback_data=f"cooking_order:{order.id}"
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="🚴 Курьер",
                        callback_data=f"assign_courier:{order.id}"
                    )
                ]
            ]
        )
        
        message_text = (
            f"🆕 Новый заказ №{order.order_number}\n\n"
            f"👤 Пользователь: {order.customer_name} (@{user.username if user.username else 'нет'})\n"
            f"📞 Телефон: {order.phone}\n"
            f"💰 Сумма: {order.total:,} сум\n"
            f"🕒 Время: {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
            f"📍 Локация: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n"
            f"💬 Комментарий: {order.comment or 'нет'}\n\n"
            f"🍽️ Заказ:\n{items_text}"
        )
        
        try:
            await bot.send_message(
                chat_id=config.SHOP_CHANNEL_ID,
                text=message_text,
                reply_markup=keyboard
            )
        except Exception as e:
            logging.error(f"Error sending to admin channel: {e}")

# ============================================================================
# Bot Handlers - Admin
# ============================================================================

@dp.message(Command("admin"))
async def cmd_admin(message: Message):
    """Admin panel"""
    if message.from_user.id not in config.ADMIN_IDS:
        await message.answer("Доступ запрещен")
        return
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🍔 Taомлар", callback_data="admin_foods")],
            [InlineKeyboardButton(text="📂 Kategoriyalar", callback_data="admin_categories")],
            [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="admin_promos")],
            [InlineKeyboardButton(text="📊 Statistika", callback_data="admin_stats")],
            [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="admin_couriers")],
            [InlineKeyboardButton(text="📦 Aktiv buyurtmalar", callback_data="admin_active_orders")],
            [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="admin_settings")]
        ]
    )
    
    await message.answer(
        "⚙️ Админ панель\n\n"
        "Выберите раздел:",
        reply_markup=keyboard
    )

@dp.callback_query(lambda c: c.data.startswith("admin_"))
async def admin_callback_handler(callback: CallbackQuery):
    """Handle admin callbacks"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    action = callback.data
    
    if action == "admin_foods":
        await show_foods_admin(callback)
    elif action == "admin_categories":
        await show_categories_admin(callback)
    elif action == "admin_promos":
        await show_promos_admin(callback)
    elif action == "admin_stats":
        await show_stats_admin(callback)
    elif action == "admin_couriers":
        await show_couriers_admin(callback)
    elif action == "admin_active_orders":
        await show_active_orders_admin(callback)
    elif action == "admin_settings":
        await show_settings_admin(callback)
    elif action == "admin_back":
        await cmd_admin(callback.message)

async def show_active_orders_admin(callback: CallbackQuery):
    """Show active orders for admin"""
    async with AsyncSessionLocal() as session:
        # Get active orders (not DELIVERED or CANCELED)
        result = await session.execute(
            select(Order)
            .where(
                Order.status.in_(["NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"])
            )
            .order_by(Order.created_at.desc())
            .limit(20)
        )
        orders = result.scalars().all()
        
        if not orders:
            await callback.message.edit_text("Активных заказов нет")
            return
        
        keyboard_buttons = []
        for order in orders:
            status_text = {
                'NEW': '🆕',
                'CONFIRMED': '✅',
                'COOKING': '🍳',
                'COURIER_ASSIGNED': '🚴',
                'OUT_FOR_DELIVERY': '📦'
            }.get(order.status, order.status)
            
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{status_text} #{order.order_number} - {order.total:,} сум",
                    callback_data=f"view_order:{order.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            f"📦 Активные заказы ({len(orders)}):",
            reply_markup=keyboard
        )

# ============================================================================
# Bot Handlers - Order Status Updates
# ============================================================================

@dp.callback_query(lambda c: c.data.startswith("confirm_order:"))
async def confirm_order(callback: CallbackQuery):
    """Confirm order"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Update order status
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if order:
            order.status = "CONFIRMED"
            order.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
            # Notify user
            try:
                await bot.send_message(
                    chat_id=order.user.tg_id,
                    text=f"✅ Ваш заказ №{order.order_number} подтвержден и готовится!"
                )
            except Exception as e:
                logging.error(f"Error notifying user: {e}")
            
            # Update admin message
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n✅ Подтвержден администратором",
                    reply_markup=None
                )
            except Exception as e:
                logging.error(f"Error editing message: {e}")
            
            await callback.answer("Заказ подтвержден")

@dp.callback_query(lambda c: c.data.startswith("cooking_order:"))
async def cooking_order(callback: CallbackQuery):
    """Mark order as cooking"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Update order status
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if order:
            order.status = "COOKING"
            order.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
            # Notify user
            try:
                await bot.send_message(
                    chat_id=order.user.tg_id,
                    text=f"🍳 Ваш заказ №{order.order_number} готовится!"
                )
            except Exception as e:
                logging.error(f"Error notifying user: {e}")
            
            # Update admin message
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n🍳 Отмечен как готовится",
                    reply_markup=None
                )
            except Exception:
                pass
            
            await callback.answer("Заказ готовится")

@dp.callback_query(lambda c: c.data.startswith("assign_courier:"))
async def assign_courier(callback: CallbackQuery):
    """Assign courier to order"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Get active couriers
        result = await session.execute(
            select(Courier).where(Courier.is_active == True)
        )
        couriers = result.scalars().all()
        
        if not couriers:
            await callback.answer("Нет активных курьеров")
            return
        
        # Get order info
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if not order:
            await callback.answer("Заказ не найден")
            return
        
        # Create courier selection keyboard
        keyboard_buttons = []
        for courier in couriers:
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"🚴 {courier.name}",
                    callback_data=f"select_courier:{order_id}:{courier.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="🔙 Назад", callback_data=f"view_order:{order_id}")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            f"Выберите курьера для заказа №{order.order_number}:",
            reply_markup=keyboard
        )

@dp.callback_query(lambda c: c.data.startswith("select_courier:"))
async def select_courier_handler(callback: CallbackQuery):
    """Handle courier selection"""
    if callback.from_user.id not in config.ADMIN_IDS:
        await callback.answer("Доступ запрещен")
        return
    
    _, order_id, courier_id = callback.data.split(":")
    order_id = int(order_id)
    courier_id = int(courier_id)
    
    async with AsyncSessionLocal() as session:
        # Update order
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if order:
            order.status = "COURIER_ASSIGNED"
            order.courier_id = courier_id
            order.updated_at = datetime.now(timezone.utc)
            
            # Get courier
            result = await session.execute(
                select(Courier).where(Courier.id == courier_id)
            )
            courier = result.scalar_one_or_none()
            
            await session.commit()
            
            # Notify courier
            if courier:
                await notify_courier(order, courier)
            
            # Update admin message
            try:
                await callback.message.edit_text(
                    callback.message.text + f"\n\n🚴 Назначен курьер: {courier.name if courier else 'Unknown'}",
                    reply_markup=None
                )
            except Exception:
                pass
            
            await callback.answer("Курьер назначен")

async def notify_courier(order: Order, courier: Courier):
    """Send order notification to courier"""
    async with AsyncSessionLocal() as session:
        # Get order items
        result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = "\n".join([
            f"• {item.name_snapshot} x{item.qty}"
            for item in items
        ])
        
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Qabul qildim",
                        callback_data=f"courier_accept:{order.id}"
                    ),
                    InlineKeyboardButton(
                        text="📦 Yetkazildi",
                        callback_data=f"courier_delivered:{order.id}"
                    )
                ]
            ]
        )
        
        message_text = (
            f"🚴 Новый заказ №{order.order_number}\n\n"
            f"👤 Клиент: {order.customer_name}\n"
            f"📞 Телефон: {order.phone}\n"
            f"💰 Сумма: {order.total:,} сум\n"
            f"📍 Локация: https://maps.google.com/?q={order.location_lat},{order.location_lng}\n"
            f"💬 Комментарий: {order.comment or 'нет'}\n\n"
            f"🍽️ Список:\n{items_text}"
        )
        
        try:
            if config.COURIER_CHANNEL_ID:
                await bot.send_message(
                    chat_id=config.COURIER_CHANNEL_ID,
                    text=message_text,
                    reply_markup=keyboard
                )
            elif courier.chat_id:
                await bot.send_message(
                    chat_id=courier.chat_id,
                    text=message_text,
                    reply_markup=keyboard
                )
        except Exception as e:
            logging.error(f"Error notifying courier: {e}")

# ============================================================================
# Bot Handlers - Courier Actions
# ============================================================================

@dp.callback_query(lambda c: c.data.startswith("courier_accept:"))
async def courier_accept(callback: CallbackQuery):
    """Courier accepts order"""
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Check if user is a courier
        result = await session.execute(
            select(Courier).where(Courier.chat_id == callback.from_user.id)
        )
        courier = result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("Вы не курьер")
            return
        
        # Update order status
        result = await session.execute(
            select(Order).where(Order.id == order_id, Order.courier_id == courier.id)
        )
        order = result.scalar_one_or_none()
        
        if order:
            order.status = "OUT_FOR_DELIVERY"
            order.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
            # Notify customer
            try:
                await bot.send_message(
                    chat_id=order.user.tg_id,
                    text=f"🚴 Ваш заказ №{order.order_number} передан курьеру и скоро будет доставлен!"
                )
            except Exception as e:
                logging.error(f"Error notifying customer: {e}")
            
            # Update courier message
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n✅ Принято курьером",
                    reply_markup=None
                )
            except Exception:
                pass
            
            await callback.answer("Заказ принят")

@dp.callback_query(lambda c: c.data.startswith("courier_delivered:"))
async def courier_delivered(callback: CallbackQuery):
    """Courier marks order as delivered"""
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Check if user is a courier
        result = await session.execute(
            select(Courier).where(Courier.chat_id == callback.from_user.id)
        )
        courier = result.scalar_one_or_none()
        
        if not courier:
            await callback.answer("Вы не курьер")
            return
        
        # Update order status
        result = await session.execute(
            select(Order).where(Order.id == order_id, Order.courier_id == courier.id)
        )
        order = result.scalar_one_or_none()
        
        if order:
            order.status = "DELIVERED"
            order.delivered_at = datetime.now(timezone.utc)
            order.updated_at = datetime.now(timezone.utc)
            await session.commit()
            
            # Notify customer
            try:
                await bot.send_message(
                    chat_id=order.user.tg_id,
                    text=f"🎉 Ваш заказ №{order.order_number} успешно доставлен! Спасибо за заказ!"
                )
            except Exception as e:
                logging.error(f"Error notifying customer: {e}")
            
            # Update courier message
            try:
                await callback.message.edit_text(
                    callback.message.text + "\n\n📦 Доставлен",
                    reply_markup=None
                )
            except Exception:
                pass
            
            await callback.answer("Заказ доставлен")

@dp.callback_query(lambda c: c.data.startswith("view_order:"))
async def view_order(callback: CallbackQuery):
    """View order details"""
    order_id = int(callback.data.split(":")[1])
    
    async with AsyncSessionLocal() as session:
        # Get order
        result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = result.scalar_one_or_none()
        
        if not order:
            await callback.answer("Заказ не найден")
            return
        
        # Get order items
        result = await session.execute(
            select(OrderItem).where(OrderItem.order_id == order.id)
        )
        items = result.scalars().all()
        
        items_text = "\n".join([
            f"• {item.name_snapshot} x{item.qty} = {item.line_total:,} сум"
            for item in items
        ])
        
        status_text = {
            'NEW': '🆕 Принят',
            'CONFIRMED': '✅ Подтвержден',
            'COOKING': '🍳 Готовится',
            'COURIER_ASSIGNED': '🚴 Курьер назначен',
            'OUT_FOR_DELIVERY': '📦 Передан курьеру',
            'DELIVERED': '🎉 Доставлен',
            'CANCELED': '❌ Отменен'
        }.get(order.status, order.status)
        
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_active_orders")]
            ]
        )
        
        await callback.message.edit_text(
            f"🆔 Заказ №{order.order_number}\n"
            f"👤 Клиент: {order.customer_name}\n"
            f"📞 Телефон: {order.phone}\n"
            f"💰 Сумма: {order.total:,} сум\n"
            f"📦 Статус: {status_text}\n"
            f"📍 Локация: {order.location_lat}, {order.location_lng}\n"
            f"🕒 Создан: {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
            f"💬 Комментарий: {order.comment or 'нет'}\n\n"
            f"🍽️ Заказ:\n{items_text}",
            reply_markup=keyboard
        )

# ============================================================================
# Additional Admin Handlers (placeholder functions)
# ============================================================================

async def show_foods_admin(callback: CallbackQuery):
    """Show foods for admin"""
    await callback.message.edit_text(
        "🍔 Управление блюдами\n\n"
        "Функционал в разработке...",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ]
        )
    )

async def show_categories_admin(callback: CallbackQuery):
    """Show categories for admin"""
    await callback.message.edit_text(
        "📂 Управление категориями\n\n"
        "Функционал в разработке...",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ]
        )
    )

async def show_promos_admin(callback: CallbackQuery):
    """Show promos for admin"""
    await callback.message.edit_text(
        "🎁 Управление промокодами\n\n"
        "Функционал в разработке...",
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ]
        )
    )

async def show_stats_admin(callback: CallbackQuery):
    """Show stats for admin"""
    async with AsyncSessionLocal() as session:
        # Get total orders count
        result = await session.execute(select(func.count(Order.id)))
        total_orders = result.scalar() or 0
        
        # Get total users count
        result = await session.execute(select(func.count(User.id)))
        total_users = result.scalar() or 0
        
        # Get today's orders count
        today = datetime.now(timezone.utc).date()
        result = await session.execute(
            select(func.count(Order.id)).where(
                func.date(Order.created_at) == today
            )
        )
        today_orders = result.scalar() or 0
        
        # Get total revenue
        result = await session.execute(
            select(func.sum(Order.total)).where(Order.status == 'DELIVERED')
        )
        total_revenue = result.scalar() or 0
        
        await callback.message.edit_text(
            f"📊 Статистика\n\n"
            f"👥 Всего пользователей: {total_users}\n"
            f"🛒 Всего заказов: {total_orders}\n"
            f"📈 Заказов сегодня: {today_orders}\n"
            f"💰 Общая выручка: {total_revenue:,} сум\n\n"
            f"Статистика обновляется в реальном времени.",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
                ]
            )
        )

async def show_couriers_admin(callback: CallbackQuery):
    """Show couriers for admin"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Courier).order_by(Courier.created_at.desc())
        )
        couriers = result.scalars().all()
        
        if not couriers:
            text = "🚴 Список курьеров пуст"
        else:
            couriers_text = []
            for courier in couriers:
                status = "✅ Активен" if courier.is_active else "❌ Неактивен"
                couriers_text.append(f"{status} - {courier.name} (ID: {courier.id})")
            
            text = "🚴 Список курьеров:\n\n" + "\n".join(couriers_text)
        
        await callback.message.edit_text(
            text,
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
                ]
            )
        )

async def show_settings_admin(callback: CallbackQuery):
    """Show settings for admin"""
    await callback.message.edit_text(
        "⚙️ Настройки\n\n"
        "Минимальная сумма заказа: 50,000 сум\n"
        "WebApp URL: https://mainsufooduz.netlify.app/webapp\n"
        "Админы: " + ", ".join(str(admin_id) for admin_id in config.ADMIN_IDS),
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ]
        )
    )

# ============================================================================
# Main Function
# ============================================================================

async def main():
    """Main async function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("aiogram").setLevel(logging.WARNING)
    
    logging.info("Starting Food Delivery Bot...")
    
    try:
        # Initialize database
        await init_db()
        logging.info("Database initialized successfully")
        
        # Set bot username
        await set_bot_username()
        logging.info(f"Bot username: @{config.get_bot_username()}")
        
        # Start polling
        logging.info("Starting bot polling...")
        await dp.start_polling(bot)
        
    except Exception as e:
        logging.error(f"Error starting bot: {e}")
    finally:
        # Cleanup
        await bot.session.close()
        if redis_client:
            await redis_client.close()
        if engine:
            await engine.dispose()
        logging.info("Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
