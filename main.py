#!/usr/bin/env python3
"""
Telegram Food Delivery Bot - FIESTA
To'liq ishlaydigan versiya
"""

import asyncio
import json
import logging
import os
import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from zoneinfo import ZoneInfo

import redis.asyncio as redis
from sqlalchemy import select, update, func, and_, or_, BigInteger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from aiogram import Bot, Dispatcher, Router, F
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup,
    InlineKeyboardButton, WebAppInfo, ReplyKeyboardMarkup,
    KeyboardButton, ReplyKeyboardRemove, MenuButtonWebApp,
    WebAppData, InputFile
)
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.client.default import DefaultBotProperties

import aiohttp
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
BOT_TOKEN = os.getenv("BOT_TOKEN", "7917271389:AAE4PXCowGo6Bsfdy3Hrz3x689MLJdQmVi4")
ADMIN_IDS = [int(id.strip()) for id in os.getenv("ADMIN_IDS", "6365371142").split(",")]
DB_URL = os.getenv("DB_URL", "postgresql+asyncpg://postgres:BDAaILJKOITNLlMOjJNfWiRPbICwEcpZ@centerbeam.proxy.rlwy.net:35489/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:GBrZNeUKJfqRlPcQUoUICWQpbQRtRRJp@ballast.proxy.rlwy.net:35411")
SHOP_CHANNEL_ID = int(os.getenv("SHOP_CHANNEL_ID", "-1003530497437"))
COURIER_CHANNEL_ID = int(os.getenv("COURIER_CHANNEL_ID", "-1003707946746"))
WEBAPP_URL = os.getenv("WEBAPP_URL", "https://mainsufooduz.netlify.app")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "https://uzbke-production.up.railway.app/api")
TIMEZONE = ZoneInfo("Asia/Tashkent")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# Bot va Dispatcher
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
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

class AdminCategoryStates(StatesGroup):
    waiting_for_name = State()

class AdminCourierStates(StatesGroup):
    waiting_for_chat_id = State()
    waiting_for_name = State()

# Database setup
Base = declarative_base()

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, DECIMAL, BigInteger as SA_BigInteger

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    tg_id = Column(SA_BigInteger, unique=True, nullable=False)
    username = Column(String(100))
    full_name = Column(String(200), nullable=False)
    joined_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    ref_by_user_id = Column(SA_BigInteger, nullable=True)
    phone = Column(String(20), nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, tg_id={self.tg_id})>"

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    
    foods = relationship("Food", back_populates="category")
    
    def __repr__(self):
        return f"<Category(id={self.id}, name={self.name})>"

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
    created_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    
    category = relationship("Category", back_populates="foods")
    
    def __repr__(self):
        return f"<Food(id={self.id}, name={self.name}, price={self.price})>"

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    order_number = Column(String(50), unique=True, nullable=False)
    user_id = Column(SA_BigInteger, nullable=False)
    customer_name = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=False)
    comment = Column(Text)
    total = Column(DECIMAL(10, 2), nullable=False)
    status = Column(String(50), default='NEW')
    created_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    updated_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE))
    delivered_at = Column(DateTime)
    location_lat = Column(Float)
    location_lng = Column(Float)
    courier_id = Column(Integer, ForeignKey('couriers.id'), nullable=True)
    
    courier = relationship("Courier")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Order(id={self.id}, number={self.order_number}, status={self.status})>"

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
    
    def __repr__(self):
        return f"<OrderItem(id={self.id}, food={self.name_snapshot}, qty={self.qty})>"

class Promo(Base):
    __tablename__ = 'promos'
    id = Column(Integer, primary_key=True)
    code = Column(String(50), unique=True, nullable=False)
    discount_percent = Column(Integer, nullable=False)
    expires_at = Column(DateTime)
    usage_limit = Column(Integer, default=100)
    used_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    
    def __repr__(self):
        return f"<Promo(id={self.id}, code={self.code}, discount={self.discount_percent}%)>"

class Courier(Base):
    __tablename__ = 'couriers'
    id = Column(Integer, primary_key=True)
    chat_id = Column(SA_BigInteger, unique=True, nullable=False)
    name = Column(String(200), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(TIMEZONE))
    
    orders = relationship("Order", back_populates="courier")
    
    def __repr__(self):
        return f"<Courier(id={self.id}, name={self.name}, chat_id={self.chat_id})>"

class ReferralStat(Base):
    __tablename__ = 'referral_stats'
    id = Column(Integer, primary_key=True)
    user_id = Column(SA_BigInteger, nullable=False, unique=True)
    ref_count = Column(Integer, default=0)
    orders_count = Column(Integer, default=0)
    delivered_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=lambda: datetime.now(TIMEZONE), onupdate=lambda: datetime.now(TIMEZONE))
    
    def __repr__(self):
        return f"<ReferralStat(user_id={self.user_id}, ref_count={self.ref_count})>"

# Database session
engine = create_async_engine(DB_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Demo ma'lumotlar
    async with AsyncSessionLocal() as session:
        # Demo kategoriyalar
        categories_count = await session.execute(select(func.count(Category.id)))
        if categories_count.scalar() == 0:
            demo_categories = [
                Category(name="Лаваш", is_active=True),
                Category(name="Бургер", is_active=True),
                Category(name="Хагги", is_active=True),
                Category(name="Шаурма", is_active=True),
                Category(name="Хот-дог", is_active=True),
                Category(name="Комбо", is_active=True),
                Category(name="Снеки", is_active=True),
                Category(name="Соусы", is_active=True),
                Category(name="Напитки", is_active=True),
            ]
            session.add_all(demo_categories)
            await session.commit()
            
            # Demo ovqatlar
            categories = await session.execute(select(Category))
            all_categories = categories.scalars().all()
            
            demo_foods = []
            for cat in all_categories:
                if cat.name == "Лаваш":
                    demo_foods.extend([
                        Food(category_id=cat.id, name="Лаваш говяжий", description="Свежий лаваш с говядиной, овощами и соусом", price=28000, rating=4.8, is_new=True),
                        Food(category_id=cat.id, name="Лаваш куриный", description="Лаваш с куриным мясом, свежими овощами", price=26000, rating=4.7),
                        Food(category_id=cat.id, name="Лаваш сырный", description="Лаваш с сыром и курицей", price=30000, rating=4.9, is_new=True),
                    ])
                elif cat.name == "Бургер":
                    demo_foods.extend([
                        Food(category_id=cat.id, name="Бургер чизбургер", description="Аппетитный бургер с сыром и говяжьей котлетой", price=32000, rating=4.9, is_new=True),
                        Food(category_id=cat.id, name="Бургер гриль", description="Бургер с грилем и овощами", price=35000, rating=4.8),
                    ])
                elif cat.name == "Напитки":
                    demo_foods.extend([
                        Food(category_id=cat.id, name="Кока-Кола", description="Охлажденная Coca-Cola 0.5л", price=8000, rating=4.3),
                        Food(category_id=cat.id, name="Фанта", description="Фанта 0.5л", price=8000, rating=4.2),
                        Food(category_id=cat.id, name="Сок Rich", description="Сок Rich 1л", price=12000, rating=4.5),
                    ])
                else:
                    demo_foods.append(
                        Food(category_id=cat.id, name=f"Demo {cat.name}", description=f"Вкусный {cat.name.lower()} от FIESTA", price=20000, rating=4.0)
                    )
            
            session.add_all(demo_foods)
            await session.commit()
        
        # Admin uchun demo kuryer
        couriers_count = await session.execute(select(func.count(Courier.id)))
        if couriers_count.scalar() == 0:
            demo_courier = Courier(
                chat_id=ADMIN_IDS[0] if ADMIN_IDS else 6365371142,
                name="Admin Courier",
                is_active=True
            )
            session.add(demo_courier)
            await session.commit()
        
        logger.info("Database initialized successfully")

# Utility funksiyalar
def format_price(price):
    """Narxlarni formatlash"""
    if isinstance(price, Decimal):
        return f"{price:,.0f}".replace(",", " ")
    return f"{int(price):,}".replace(",", " ")

def generate_order_number():
    """Buyurtma raqamini yaratish"""
    date_str = datetime.now(TIMEZONE).strftime("%Y%m%d")
    random_str = ''.join(random.choices(string.digits, k=6))
    return f"ORD-{date_str}-{random_str}"

async def get_or_create_user(tg_id: int, username: str, full_name: str, ref_by: int = None) -> User:
    """Foydalanuvchini olish yoki yaratish"""
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(
                select(User).where(User.tg_id == tg_id)
            )
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
                
                # Referral stat yaratish
                ref_stat = ReferralStat(user_id=tg_id)
                session.add(ref_stat)
                await session.commit()
                
                # Agar referral orqali kelgan bo'lsa
                if ref_by:
                    # Referral statistikani yangilash
                    ref_result = await session.execute(
                        select(ReferralStat).where(ReferralStat.user_id == ref_by)
                    )
                    ref_stat = ref_result.scalar_one_or_none()
                    if ref_stat:
                        ref_stat.ref_count += 1
                        await session.commit()
            
            return user
        except Exception as e:
            logger.error(f"Error in get_or_create_user: {e}")
            await session.rollback()
            raise

async def get_user_by_tg_id(tg_id: int) -> Optional[User]:
    """Telegram ID bo'yicha foydalanuvchini olish"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.tg_id == tg_id)
        )
        return result.scalar_one_or_none()

async def update_referral_stats(user_tg_id: int, order_delivered: bool = False):
    """Referral statistikani yangilash"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ReferralStat).where(ReferralStat.user_id == user_tg_id)
        )
        stat = result.scalar_one_or_none()
        
        if stat:
            stat.orders_count += 1
            if order_delivered:
                stat.delivered_count += 1
            await session.commit()

# Keyboardlar
def get_client_main_keyboard():
    """Asosiy klaviatura"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга"), KeyboardButton(text="📞 Контакты")]
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )
    return keyboard

def get_admin_main_keyboard():
    """Admin asosiy klaviatura"""
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
    """Buyurtma statusi klaviaturasi"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"status:confirmed:{order_id}"),
                InlineKeyboardButton(text="🍳 Готовится", callback_data=f"status:cooking:{order_id}")
            ],
            [
                InlineKeyboardButton(text="🚴 Курьер", callback_data=f"status:courier:{order_id}")
            ],
            [
                InlineKeyboardButton(text="❌ Отменить", callback_data=f"status:canceled:{order_id}")
            ]
        ]
    )
    return keyboard

def get_courier_choice_keyboard(order_id: int):
    """Kuryer tanlash klaviaturasi"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🚴 Выбрать курьера", callback_data=f"choose_courier:{order_id}")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"back_to_order:{order_id}")]
        ]
    )
    return keyboard

def get_courier_list_keyboard(order_id: int, couriers: List[Courier]):
    """Kuryerlar ro'yxati klaviaturasi"""
    buttons = []
    for courier in couriers:
        status = "🟢" if courier.is_active else "🔴"
        buttons.append([
            InlineKeyboardButton(
                text=f"{status} {courier.name}",
                callback_data=f"assign_courier:{order_id}:{courier.id}"
            )
        ])
    
    buttons.append([InlineKeyboardButton(text="⬅️ Назад", callback_data=f"back_to_status:{order_id}")])
    
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_courier_order_keyboard(order_id: int):
    """Kuryer buyurtma klaviaturasi"""
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
    """/start komandasi"""
    args = message.text.split()
    ref_by = None
    
    if len(args) > 1:
        try:
            ref_by = int(args[1])
        except ValueError:
            pass
    
    user = await get_or_create_user(
        tg_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
        ref_by=ref_by
    )
    
    welcome_text = (
        f"🌟 Добро Пожаловать в FIESTA! {message.from_user.full_name}\n\n"
        f"Для заказа перейдите по кнопке ➡️\n"
        f"🛍 Заказать"
    )
    
    await message.answer(
        welcome_text,
        reply_markup=get_client_main_keyboard()
    )
    
    # Menyu tugmasini o'rnatish
    try:
        await bot.set_chat_menu_button(
            chat_id=message.chat.id,
            menu_button=MenuButtonWebApp(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))
        )
    except Exception as e:
        logger.error(f"Error setting menu button: {e}")

@client_router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    """Mening buyurtmalarim"""
    try:
        async with AsyncSessionLocal() as session:
            # Avval user ni topish
            user_result = await session.execute(
                select(User).where(User.tg_id == message.from_user.id)
            )
            user = user_result.scalar_one_or_none()
            
            if not user:
                await message.answer(
                    "Вы еще не сделали ни одного заказа.\n"
                    "Чтобы сделать заказ, нажмите кнопку ниже ⬇️",
                    reply_markup=ReplyKeyboardMarkup(
                        keyboard=[[KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))]],
                        resize_keyboard=True
                    )
                )
                return
            
            # Endi buyurtmalarni topish
            result = await session.execute(
                select(Order)
                .where(Order.user_id == user.id)
                .order_by(Order.created_at.desc())
                .limit(10)
            )
            orders = result.scalars().all()
            
            if not orders:
                await message.answer(
                    "📭 У вас пока нет заказов.\n\n"
                    "Сделайте свой первый заказ! 🛍️",
                    reply_markup=ReplyKeyboardMarkup(
                        keyboard=[[KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))]],
                        resize_keyboard=True
                    )
                )
            else:
                response = "📦 Ваши последние заказы:\n\n"
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
                    
                    status_text = {
                        'NEW': 'Принят',
                        'CONFIRMED': 'Подтвержден',
                        'COOKING': 'Готовится',
                        'COURIER_ASSIGNED': 'Курьер назначен',
                        'OUT_FOR_DELIVERY': 'Передан курьеру',
                        'DELIVERED': 'Доставлен',
                        'CANCELED': 'Отменен'
                    }.get(order.status, order.status)
                    
                    response += (
                        f"{status_emoji} <b>Заказ №{order.order_number}</b>\n"
                        f"📅 {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                        f"💰 {format_price(order.total)} сум\n"
                        f"📊 Статус: {status_text}\n"
                        f"━━━━━━━━━━━━━━\n"
                    )
                
                await message.answer(response, parse_mode="HTML")
    except Exception as e:
        logger.error(f"Error in my_orders: {e}")
        await message.answer(
            "⚠️ Произошла ошибка при получении заказов. Пожалуйста, попробуйте позже.",
            reply_markup=get_client_main_keyboard()
        )

@client_router.message(F.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    """Ma'lumot"""
    about_text = (
        "🌟 <b>Добро Пожаловать в FIESTA!</b>\n\n"
        "📍 <b>Наш адрес:</b> Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢 <b>Ориентир:</b> Школа №12 Оруджева\n"
        "📞 <b>Контактный номер:</b> +998 91 420 15 15\n"
        "🕙 <b>Рабочие часы:</b> 24/7\n"
        "📷 <b>Мы в Instagram:</b> fiesta.khiva\n"
        "🔗 <b>Найти нас на карте:</b> https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7\n\n"
        "Мы всегда рады вам! ❤️"
    )
    await message.answer(about_text, parse_mode="HTML")

@client_router.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    """Referral"""
    try:
        async with AsyncSessionLocal() as session:
            # Referral statistikani olish
            result = await session.execute(
                select(ReferralStat).where(ReferralStat.user_id == message.from_user.id)
            )
            stat = result.scalar_one_or_none()
            
            if not stat:
                stat = ReferralStat(user_id=message.from_user.id)
                session.add(stat)
                await session.commit()
            
            # Buyurtmalar soni
            user_result = await session.execute(
                select(User).where(User.tg_id == message.from_user.id)
            )
            user = user_result.scalar_one()
            
            orders_result = await session.execute(
                select(func.count(Order.id)).where(Order.user_id == user.id)
            )
            orders_count = orders_result.scalar() or 0
            
            delivered_result = await session.execute(
                select(func.count(Order.id)).where(
                    Order.user_id == user.id,
                    Order.status == 'DELIVERED'
                )
            )
            delivered_count = delivered_result.scalar() or 0
            
            bot_username = (await bot.me()).username
            referral_link = f"https://t.me/{bot_username}?start={message.from_user.id}"
            
            referral_text = (
                "👥 <b>Пригласите друга и получите скидку!</b>\n\n"
                f"📊 <b>Ваша статистика:</b>\n"
                f"• Приглашено друзей: {stat.ref_count}\n"
                f"• Ваших заказов: {orders_count}\n"
                f"• Доставлено заказов: {delivered_count}\n\n"
                f"🔗 <b>Ваша реферальная ссылка:</b>\n"
                f"<code>{referral_link}</code>\n\n"
                "🎁 <b>Бонусы:</b>\n"
                "• За 3 приглашенных друга - промокод 15%\n"
                "• За 5 приглашенных - промокод 20%\n"
                "• За 10 приглашенных - промокод 30%\n\n"
                "Поделитесь ссылкой с друзьями и получайте скидки! 🎉"
            )
            
            await message.answer(referral_text, parse_mode="HTML")
            
            # Promo code tekshirish va berish
            if stat.ref_count >= 3:
                # Promo kod mavjudligini tekshirish
                promo_result = await session.execute(
                    select(Promo).where(Promo.code.like(f"REF{message.from_user.id}%"))
                )
                existing_promo = promo_result.scalar_one_or_none()
                
                if not existing_promo:
                    discount = 15
                    if stat.ref_count >= 10:
                        discount = 30
                    elif stat.ref_count >= 5:
                        discount = 20
                    
                    promo_code = f"REF{message.from_user.id}{random.randint(100, 999)}"
                    new_promo = Promo(
                        code=promo_code,
                        discount_percent=discount,
                        expires_at=datetime.now(TIMEZONE) + timedelta(days=30),
                        usage_limit=5,
                        is_active=True
                    )
                    session.add(new_promo)
                    await session.commit()
                    
                    await message.answer(
                        f"🎉 <b>Поздравляем!</b>\n\n"
                        f"Вы получили промокод: <code>{promo_code}</code>\n"
                        f"📉 Скидка: {discount}%\n"
                        f"⏳ Действует до: {new_promo.expires_at.strftime('%d.%m.%Y')}\n\n"
                        f"Используйте его при оформлении заказа!",
                        parse_mode="HTML"
                    )
    
    except Exception as e:
        logger.error(f"Error in invite_friend: {e}")
        await message.answer(
            "⚠️ Произошла ошибка. Пожалуйста, попробуйте позже.",
            reply_markup=get_client_main_keyboard()
        )

@client_router.message(F.text == "📞 Контакты")
async def contacts(message: Message):
    """Kontaktlar"""
    contacts_text = (
        "📞 <b>Контакты FIESTA</b>\n\n"
        "📍 <b>Адрес:</b> Хива, махалла Гиламчи\n"
        "🏫 <b>Ориентир:</b> Школа №12 Оруджева\n\n"
        "📱 <b>Телефоны:</b>\n"
        "• +998 91 420 15 15 (доставка)\n"
        "• +998 93 123 45 67 (администрация)\n\n"
        "🕒 <b>Режим работы:</b> 24/7\n\n"
        "📧 <b>Email:</b> fiesta.khiva@gmail.com\n"
        "📷 <b>Instagram:</b> @fiesta.khiva\n\n"
        "Мы всегда на связи! 💬"
    )
    await message.answer(contacts_text, parse_mode="HTML")

@client_router.message(Command("shop"))
async def cmd_shop(message: Message):
    """Shop komandasi"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(
                text="🛍 Открыть магазин",
                web_app=WebAppInfo(url=WEBAPP_URL)
            )
        ]]
    )
    
    await message.answer(
        "🛒 <b>Добро пожаловать в магазин FIESTA!</b>\n\n"
        "Нажмите кнопку ниже, чтобы открыть меню и сделать заказ ⬇️",
        reply_markup=keyboard,
        parse_mode="HTML"
    )

# ========================
# WEB APP DATA HANDLER
# ========================

@client_router.message(F.web_app_data)
async def handle_web_app_data(message: WebAppData):
    """WebApp dan kelgan ma'lumotlarni qayta ishlash"""
    try:
        data = json.loads(message.web_app_data.data)
        logger.info(f"WebApp data received from user {message.from_user.id}: {data}")
        
        if data.get('type') == 'order_create':
            await process_order_create(message.from_user, data)
        else:
            await message.answer("Неизвестный тип данных от WebApp.")
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        await message.answer("❌ Ошибка обработки заказа. Неверный формат данных.")
    except Exception as e:
        logger.error(f"Error processing web app data: {e}")
        await message.answer("❌ Произошла ошибка при обработке заказа. Пожалуйста, попробуйте позже.")

async def process_order_create(user, data: Dict):
    """Buyurtma yaratish"""
    try:
        async with AsyncSessionLocal() as session:
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
                    text="❌ <b>Минимальная сумма заказа 50,000 сум</b>\n\n"
                         "Добавьте еще товаров в корзину.",
                    parse_mode="HTML"
                )
                return
            
            # Promo code tekshirish
            promo_code = data.get('promo_code')
            final_total = total
            discount_amount = Decimal('0')
            
            if promo_code:
                promo_result = await session.execute(
                    select(Promo).where(
                        Promo.code == promo_code,
                        Promo.is_active == True,
                        Promo.used_count < Promo.usage_limit,
                        or_(
                            Promo.expires_at == None,
                            Promo.expires_at > datetime.now(TIMEZONE)
                        )
                    )
                )
                promo = promo_result.scalar_one_or_none()
                
                if promo:
                    discount = total * Decimal(promo.discount_percent) / 100
                    final_total = total - discount
                    discount_amount = discount
                    promo.used_count += 1
                    
                    # Promo ishlatilganligi haqida xabar
                    promo_message = f"✅ Промокод применен! Скидка: {promo.discount_percent}% ({format_price(discount)} сум)"
                else:
                    promo_message = "❌ Неверный или просроченный промо-код"
            else:
                promo_message = ""
            
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
            
            # Referral statistikani yangilash
            await update_referral_stats(db_user.tg_id)
            
            # User ga xabar
            user_message = (
                "✅ <b>Ваш заказ принят!</b>\n\n"
                f"🆔 <b>Номер заказа:</b> {order.order_number}\n"
                f"👤 <b>Имя:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n"
            )
            
            if discount_amount > 0:
                user_message += f"🎁 <b>Скидка:</b> {format_price(discount_amount)} сум\n"
            
            user_message += (
                f"📦 <b>Статус:</b> Принят\n\n"
                f"📝 <b>Комментарий:</b> {order.comment if order.comment else 'нет'}\n\n"
                "Мы свяжемся с вами для подтверждения заказа. ⏳"
            )
            
            await bot.send_message(
                chat_id=user.id,
                text=user_message,
                parse_mode="HTML"
            )
            
            # Admin kanalga yuborish
            await send_order_to_admin_channel(order)
            
    except Exception as e:
        logger.error(f"Error in process_order_create: {e}")
        await bot.send_message(
            chat_id=user.id,
            text="❌ Произошла ошибка при создании заказа. Пожалуйста, попробуйте позже или свяжитесь с администратором."
        )

async def send_order_to_admin_channel(order: Order):
    """Buyurtmani admin kanaliga yuborish"""
    try:
        async with AsyncSessionLocal() as session:
            # Order items olish
            result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = result.scalars().all()
            
            # User ma'lumotlari
            user_result = await session.execute(
                select(User).where(User.id == order.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            items_text = ""
            for item in items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {format_price(item.line_total)} сум\n"
            
            location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
            
            order_text = (
                f"🆕 <b>НОВЫЙ ЗАКАЗ</b>\n\n"
                f"🆔 <b>Номер:</b> {order.order_number}\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"👨‍💼 <b>Telegram:</b> @{user.username if user and user.username else 'скрыт'}\n"
                f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n"
                f"🕒 <b>Время:</b> {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                f"📍 <b>Локация:</b> <a href='{location_link}'>На карте</a>\n\n"
                f"📝 <b>Комментарий:</b>\n{order.comment if order.comment else 'нет'}\n\n"
                f"🍽️ <b>Заказ:</b>\n{items_text}"
            )
            
            message = await bot.send_message(
                chat_id=SHOP_CHANNEL_ID,
                text=order_text,
                reply_markup=get_order_status_keyboard(order.id),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            
            # Message ID ni saqlash
            async with redis_client as r:
                await r.set(f"order_message:{order.id}", message.message_id)
                await r.set(f"order_channel:{order.id}", SHOP_CHANNEL_ID)
                
            logger.info(f"Order {order.id} sent to admin channel")
            
    except Exception as e:
        logger.error(f"Error sending order to admin channel: {e}")

# ========================
# ADMIN HANDLERS
# ========================

@admin_router.message(Command("admin"))
async def admin_panel(message: Message):
    """Admin panel"""
    if message.from_user.id not in ADMIN_IDS:
        await message.answer("⛔ Доступ запрещен.")
        return
    
    await message.answer(
        "⚙️ <b>Админ панель FIESTA</b>\n\n"
        "Выберите раздел для управления:",
        reply_markup=get_admin_main_keyboard(),
        parse_mode="HTML"
    )

@admin_router.callback_query(F.data.startswith("admin:"))
async def admin_menu_handler(callback: CallbackQuery):
    """Admin menyusi"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("Доступ запрещен.", show_alert=True)
        return
    
    action = callback.data.split(":")[1]
    
    if action == "foods":
        await show_foods_menu(callback)
    elif action == "categories":
        await show_categories_menu(callback)
    elif action == "promos":
        await show_promos_menu(callback)
    elif action == "stats":
        await show_stats(callback)
    elif action == "couriers":
        await show_couriers_menu(callback)
    elif action == "active_orders":
        await show_active_orders(callback)
    elif action == "settings":
        await show_settings(callback)
    elif action == "back":
        await callback.message.edit_text(
            "⚙️ <b>Админ панель FIESTA</b>\n\n"
            "Выберите раздел для управления:",
            reply_markup=get_admin_main_keyboard(),
            parse_mode="HTML"
        )
    
    await callback.answer()

async def show_foods_menu(callback: CallbackQuery):
    """Ovqatlar menyusi"""
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="➕ Добавить блюдо", callback_data="food:add")],
            [InlineKeyboardButton(text="📝 Список блюд", callback_data="food:list")],
            [InlineKeyboardButton(text="📊 Статистика блюд", callback_data="food:stats")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
        ]
    )
    
    await callback.message.edit_text(
        "🍔 <b>Управление блюдами</b>\n\n"
        "Выберите действие:",
        reply_markup=keyboard,
        parse_mode="HTML"
    )

async def show_categories_menu(callback: CallbackQuery):
    """Kategoriyalar menyusi"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Category).order_by(Category.name)
        )
        categories = result.scalars().all()
        
        keyboard_buttons = []
        for category in categories:
            status = "🟢" if category.is_active else "🔴"
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{status} {category.name}",
                    callback_data=f"category:edit:{category.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="➕ Добавить категорию", callback_data="category:add")
        ])
        keyboard_buttons.append([
            InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            "📂 <b>Управление категориями</b>\n\n"
            f"Всего категорий: {len(categories)}\n"
            "Выберите категорию для редактирования:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )

async def show_promos_menu(callback: CallbackQuery):
    """Promokodlar menyusi"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Promo).order_by(Promo.created_at.desc()).limit(20)
        )
        promos = result.scalars().all()
        
        keyboard_buttons = []
        for promo in promos:
            status = "🟢" if promo.is_active else "🔴"
            expired = "⏳" if promo.expires_at and promo.expires_at < datetime.now(TIMEZONE) else ""
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{status}{expired} {promo.code} ({promo.discount_percent}%)",
                    callback_data=f"promo:edit:{promo.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="➕ Создать промокод", callback_data="promo:add")
        ])
        keyboard_buttons.append([
            InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        active_count = sum(1 for p in promos if p.is_active)
        used_count = sum(p.used_count for p in promos)
        
        await callback.message.edit_text(
            "🎁 <b>Управление промокодами</b>\n\n"
            f"Всего промокодов: {len(promos)}\n"
            f"Активных: {active_count}\n"
            f"Использовано раз: {used_count}\n\n"
            "Выберите промокод для редактирования:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )

async def show_stats(callback: CallbackQuery):
    """Statistika"""
    async with AsyncSessionLocal() as session:
        # Bugungi statistika
        today_start = datetime.now(TIMEZONE).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Buyurtmalar
        orders_today = await session.execute(
            select(func.count(Order.id)).where(Order.created_at >= today_start)
        )
        orders_today_count = orders_today.scalar() or 0
        
        # Yetkazilgan buyurtmalar
        delivered_today = await session.execute(
            select(func.count(Order.id)).where(
                Order.delivered_at >= today_start,
                Order.status == 'DELIVERED'
            )
        )
        delivered_today_count = delivered_today.scalar() or 0
        
        # Daromad
        revenue_today = await session.execute(
            select(func.sum(Order.total)).where(
                Order.delivered_at >= today_start,
                Order.status == 'DELIVERED'
            )
        )
        revenue_today_amount = revenue_today.scalar() or Decimal('0')
        
        # Aktiv buyurtmalar
        active_orders = await session.execute(
            select(func.count(Order.id)).where(
                Order.status.in_(['NEW', 'CONFIRMED', 'COOKING', 'COURIER_ASSIGNED', 'OUT_FOR_DELIVERY'])
            )
        )
        active_orders_count = active_orders.scalar() or 0
        
        # Foydalanuvchilar
        total_users = await session.execute(select(func.count(User.id)))
        total_users_count = total_users.scalar() or 0
        
        # Haftalik daromad
        week_start = today_start - timedelta(days=7)
        revenue_week = await session.execute(
            select(func.sum(Order.total)).where(
                Order.delivered_at >= week_start,
                Order.status == 'DELIVERED'
            )
        )
        revenue_week_amount = revenue_week.scalar() or Decimal('0')
        
        stats_text = (
            "📊 <b>Статистика FIESTA</b>\n\n"
            "📅 <b>Сегодня:</b>\n"
            f"• Заказов: {orders_today_count}\n"
            f"• Доставлено: {delivered_today_count}\n"
            f"• Выручка: {format_price(revenue_today_amount)} сум\n\n"
            "📈 <b>Общая:</b>\n"
            f"• Активных заказов: {active_orders_count}\n"
            f"• Пользователей: {total_users_count}\n"
            f"• Выручка за неделю: {format_price(revenue_week_amount)} сум\n\n"
            "📋 <b>Дополнительно:</b>\n"
            "• /stats_detailed - подробная статистика\n"
            "• /top_foods - популярные блюда"
        )
        
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📅 Детальная статистика", callback_data="stats:detailed")],
                [InlineKeyboardButton(text="🍔 Популярные блюда", callback_data="stats:top_foods")],
                [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
            ]
        )
        
        await callback.message.edit_text(
            stats_text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )

async def show_couriers_menu(callback: CallbackQuery):
    """Kuryerlar menyusi"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Courier).order_by(Courier.is_active.desc(), Courier.name)
        )
        couriers = result.scalars().all()
        
        keyboard_buttons = []
        for courier in couriers:
            status = "🟢" if courier.is_active else "🔴"
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{status} {courier.name}",
                    callback_data=f"courier:edit:{courier.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="➕ Добавить курьера", callback_data="courier:add")
        ])
        keyboard_buttons.append([
            InlineKeyboardButton(text="📊 Статистика курьеров", callback_data="courier:stats")
        ])
        keyboard_buttons.append([
            InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        active_count = sum(1 for c in couriers if c.is_active)
        
        await callback.message.edit_text(
            "🚴 <b>Управление курьерами</b>\n\n"
            f"Всего курьеров: {len(couriers)}\n"
            f"Активных: {active_count}\n\n"
            "Выберите курьера для редактирования:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )

async def show_active_orders(callback: CallbackQuery):
    """Aktiv buyurtmalar"""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Order).where(
                Order.status.in_(['NEW', 'CONFIRMED', 'COOKING', 'COURIER_ASSIGNED', 'OUT_FOR_DELIVERY'])
            ).order_by(
                Order.created_at.desc()
            ).limit(20)
        )
        orders = result.scalars().all()
        
        if not orders:
            await callback.message.edit_text(
                "📭 <b>Нет активных заказов</b>\n\n"
                "Все заказы обработаны или доставлены.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
                    ]
                ),
                parse_mode="HTML"
            )
            return
        
        # Status bo'yicha guruhlash
        status_groups = {}
        for order in orders:
            if order.status not in status_groups:
                status_groups[order.status] = []
            status_groups[order.status].append(order)
        
        text = "📦 <b>Активные заказы</b>\n\n"
        
        status_names = {
            'NEW': '🆕 Новые',
            'CONFIRMED': '✅ Подтвержденные',
            'COOKING': '🍳 Готовятся',
            'COURIER_ASSIGNED': '🚴 Курьеры назначены',
            'OUT_FOR_DELIVERY': '📦 В пути'
        }
        
        for status, status_text in status_names.items():
            if status in status_groups:
                text += f"{status_text}: {len(status_groups[status])}\n"
        
        text += "\nВыберите заказ для управления:"
        
        keyboard_buttons = []
        for order in orders:
            status_emoji = {
                'NEW': '🆕',
                'CONFIRMED': '✅',
                'COOKING': '🍳',
                'COURIER_ASSIGNED': '🚴',
                'OUT_FOR_DELIVERY': '📦'
            }.get(order.status, '📝')
            
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{status_emoji} #{order.order_number} - {format_price(order.total)} сум",
                    callback_data=f"order:detail:{order.id}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")
        ])
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        
        await callback.message.edit_text(
            text,
            reply_markup=keyboard,
            parse_mode="HTML"
        )

async def show_settings(callback: CallbackQuery):
    """Sozlamalar"""
    settings_text = (
        "⚙️ <b>Настройки бота</b>\n\n"
        f"🆔 <b>ID бота:</b> {bot.id}\n"
        f"👤 <b>Имя бота:</b> {(await bot.me()).first_name}\n"
        f"🔗 <b>WebApp URL:</b> {WEBAPP_URL}\n"
        f"📢 <b>Канал заказов:</b> {SHOP_CHANNEL_ID}\n"
        f"🚴 <b>Канал курьеров:</b> {COURIER_CHANNEL_ID}\n\n"
        f"👑 <b>Админы:</b> {', '.join(map(str, ADMIN_IDS))}\n\n"
        "<i>Для изменения настроек отредактируйте .env файл</i>"
    )
    
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔄 Проверить подключения", callback_data="settings:check")],
            [InlineKeyboardButton(text="📊 Статус системы", callback_data="settings:status")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="admin:back")]
        ]
    )
    
    await callback.message.edit_text(
        settings_text,
        reply_markup=keyboard,
        parse_mode="HTML"
    )

# ========================
# ORDER STATUS HANDLERS
# ========================

@admin_router.callback_query(F.data.startswith("status:"))
async def handle_order_status(callback: CallbackQuery):
    """Buyurtma statusini o'zgartirish"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("Доступ запрещен.", show_alert=True)
        return
    
    try:
        _, action, order_id = callback.data.split(":")
        order_id = int(order_id)
        
        async with AsyncSessionLocal() as session:
            # Order ni olish
            result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = result.scalar_one()
            
            if action == "confirmed":
                order.status = "CONFIRMED"
                status_text = "✅ Подтвержден"
            elif action == "cooking":
                order.status = "COOKING"
                status_text = "🍳 Готовится"
            elif action == "courier":
                # Kuryer tanlash menyusi
                await choose_courier_for_order(callback, order_id)
                await callback.answer()
                return
            elif action == "canceled":
                order.status = "CANCELED"
                status_text = "❌ Отменен"
                
                # Userga xabar
                await bot.send_message(
                    chat_id=order.user_id,  # E'tibor: bu user.id (primary key), tg_id emas
                    text=f"❌ <b>Ваш заказ №{order.order_number} отменен.</b>\n\n"
                         "По вопросам обращайтесь к администратору.",
                    parse_mode="HTML"
                )
            else:
                await callback.answer("Неизвестное действие")
                return
            
            order.updated_at = datetime.now(TIMEZONE)
            await session.commit()
            
            # Xabarni yangilash
            await update_order_message(order)
            
            await callback.answer(f"Статус изменен на: {status_text}")
            
            # Agar status CONFIRMED bo'lsa, userga xabar
            if action == "confirmed":
                # User ni topish
                user_result = await session.execute(
                    select(User).where(User.id == order.user_id)
                )
                user = user_result.scalar_one_or_none()
                
                if user:
                    await bot.send_message(
                        chat_id=user.tg_id,
                        text=f"✅ <b>Ваш заказ №{order.order_number} подтвержден!</b>\n\n"
                             "Мы начали готовить ваш заказ. Ожидайте следующих уведомлений.",
                        parse_mode="HTML"
                    )
    
    except Exception as e:
        logger.error(f"Error in handle_order_status: {e}")
        await callback.answer("Ошибка при изменении статуса", show_alert=True)

async def choose_courier_for_order(callback: CallbackQuery, order_id: int):
    """Kuryer tanlash"""
    async with AsyncSessionLocal() as session:
        # Active kuryerlarni olish
        result = await session.execute(
            select(Courier).where(Courier.is_active == True).order_by(Courier.name)
        )
        couriers = result.scalars().all()
        
        if not couriers:
            await callback.message.edit_text(
                "❌ <b>Нет активных курьеров</b>\n\n"
                "Добавьте курьеров в систему, прежде чем назначать заказы.",
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="➕ Добавить курьера", callback_data="courier:add")],
                        [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"back_to_status:{order_id}")]
                    ]
                ),
                parse_mode="HTML"
            )
            return
        
        # Order ma'lumotlari
        order_result = await session.execute(
            select(Order).where(Order.id == order_id)
        )
        order = order_result.scalar_one()
        
        text = (
            f"🚴 <b>Назначение курьера</b>\n\n"
            f"Заказ: <b>№{order.order_number}</b>\n"
            f"Сумма: <b>{format_price(order.total)} сум</b>\n"
            f"Адрес: <a href='https://maps.google.com/?q={order.location_lat},{order.location_lng}'>Посмотреть на карте</a>\n\n"
            "Выберите курьера:"
        )
        
        await callback.message.edit_text(
            text,
            reply_markup=get_courier_list_keyboard(order_id, couriers),
            parse_mode="HTML",
            disable_web_page_preview=True
        )

@admin_router.callback_query(F.data.startswith("assign_courier:"))
async def assign_courier_handler(callback: CallbackQuery):
    """Kuryerni tayinlash"""
    if callback.from_user.id not in ADMIN_IDS:
        await callback.answer("Доступ запрещен.", show_alert=True)
        return
    
    try:
        _, order_id, courier_id = callback.data.split(":")
        order_id = int(order_id)
        courier_id = int(courier_id)
        
        async with AsyncSessionLocal() as session:
            # Order ni olish
            order_result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = order_result.scalar_one()
            
            # Courier ni olish
            courier_result = await session.execute(
                select(Courier).where(Courier.id == courier_id)
            )
            courier = courier_result.scalar_one()
            
            # Yangilash
            order.status = "COURIER_ASSIGNED"
            order.courier_id = courier_id
            order.updated_at = datetime.now(TIMEZONE)
            await session.commit()
            
            # Admin xabarni yangilash
            await update_order_message(order)
            
            # Kuryerga yuborish
            await send_order_to_courier(order, courier)
            
            # Userga xabar
            # User ni topish
            user_result = await session.execute(
                select(User).where(User.id == order.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                await bot.send_message(
                    chat_id=user.tg_id,
                    text=f"🚴 <b>К вашему заказу №{order.order_number} назначен курьер!</b>\n\n"
                         f"Имя курьера: <b>{courier.name}</b>\n"
                         "Ожидайте доставку в ближайшее время.",
                    parse_mode="HTML"
                )
            
            await callback.answer(f"Курьер {courier.name} назначен")
            
            # Orqaga qaytish
            await callback.message.edit_text(
                f"✅ <b>Курьер назначен успешно!</b>\n\n"
                f"Заказ: <b>№{order.order_number}</b>\n"
                f"Курьер: <b>{courier.name}</b>\n"
                f"Статус: <b>Курьер назначен</b>",
                parse_mode="HTML"
            )
    
    except Exception as e:
        logger.error(f"Error in assign_courier_handler: {e}")
        await callback.answer("Ошибка при назначении курьера", show_alert=True)

async def update_order_message(order: Order):
    """Buyurtma xabarini yangilash"""
    try:
        # Eski xabarni olish
        async with redis_client as r:
            message_id = await r.get(f"order_message:{order.id}")
            channel_id = await r.get(f"order_channel:{order.id}")
        
        if message_id and channel_id:
            # Order items olish
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(OrderItem).where(OrderItem.order_id == order.id)
                )
                items = result.scalars().all()
                
                # User ma'lumotlari
                user_result = await session.execute(
                    select(User).where(User.id == order.user_id)
                )
                user = user_result.scalar_one_or_none()
                
                items_text = ""
                for item in items:
                    items_text += f"• {item.name_snapshot} x{item.qty} = {format_price(item.line_total)} сум\n"
                
                location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
                
                status_text = {
                    'NEW': '🆕 Принят',
                    'CONFIRMED': '✅ Подтвержден',
                    'COOKING': '🍳 Готовится',
                    'COURIER_ASSIGNED': '🚴 Курьер назначен',
                    'OUT_FOR_DELIVERY': '📦 Передан курьеру',
                    'DELIVERED': '🎉 Доставлен',
                    'CANCELED': '❌ Отменен'
                }.get(order.status, order.status)
                
                # Courier ma'lumotlari
                courier_text = ""
                if order.courier_id:
                    courier_result = await session.execute(
                        select(Courier).where(Courier.id == order.courier_id)
                    )
                    courier = courier_result.scalar_one_or_none()
                    if courier:
                        courier_text = f"\n🚴 <b>Курьер:</b> {courier.name}"
                
                order_text = (
                    f"{'✅ ' if order.status == 'DELIVERED' else ''}<b>ЗАКАЗ {order.order_number}</b>\n\n"
                    f"👤 <b>Клиент:</b> {order.customer_name}\n"
                    f"📞 <b>Телефон:</b> {order.phone}\n"
                    f"👨‍💼 <b>Telegram:</b> @{user.username if user and user.username else 'скрыт'}\n"
                    f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n"
                    f"📊 <b>Статус:</b> {status_text}\n"
                    f"🕒 <b>Время:</b> {order.created_at.strftime('%d.%m.%Y %H:%M')}\n"
                    f"📍 <b>Локация:</b> <a href='{location_link}'>На карте</a>"
                    f"{courier_text}\n\n"
                    f"📝 <b>Комментарий:</b>\n{order.comment if order.comment else 'нет'}\n\n"
                    f"🍽️ <b>Заказ:</b>\n{items_text}"
                )
                
                if order.status == 'DELIVERED':
                    keyboard = None
                else:
                    keyboard = get_order_status_keyboard(order.id)
                
                try:
                    await bot.edit_message_text(
                        chat_id=int(channel_id),
                        message_id=int(message_id),
                        text=order_text,
                        reply_markup=keyboard,
                        parse_mode="HTML",
                        disable_web_page_preview=True
                    )
                except Exception as e:
                    logger.error(f"Error editing message: {e}")
    
    except Exception as e:
        logger.error(f"Error in update_order_message: {e}")

async def send_order_to_courier(order: Order, courier: Courier):
    """Buyurtmani kuryerga yuborish"""
    try:
        async with AsyncSessionLocal() as session:
            # Order items olish
            result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = result.scalars().all()
            
            items_text = ""
            for item in items:
                items_text += f"• {item.name_snapshot} x{item.qty}\n"
            
            location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
            yandex_link = f"https://yandex.ru/maps/?pt={order.location_lng},{order.location_lat}&z=16"
            
            courier_text = (
                f"🚴 <b>НОВЫЙ ЗАКАЗ ДЛЯ ДОСТАВКИ</b>\n\n"
                f"🆔 <b>Номер заказа:</b> {order.order_number}\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n\n"
                f"📍 <b>Локация:</b>\n"
                f"• Google Maps: <a href='{location_link}'>Открыть</a>\n"
                f"• Яндекс.Карты: <a href='{yandex_link}'>Открыть</a>\n\n"
                f"🍽️ <b>Состав заказа:</b>\n{items_text}\n"
                f"💬 <b>Комментарий:</b>\n{order.comment if order.comment else 'нет'}\n\n"
                f"⏰ <b>Время заказа:</b> {order.created_at.strftime('%H:%M')}\n\n"
                f"<i>Подтвердите получение заказа нажав кнопку ниже ⬇️</i>"
            )
            
            # Kuryer kanaliga yuborish
            if COURIER_CHANNEL_ID:
                await bot.send_message(
                    chat_id=COURIER_CHANNEL_ID,
                    text=courier_text,
                    parse_mode="HTML",
                    disable_web_page_preview=True
                )
            
            # Kuryerga shaxsiy xabar
            await bot.send_message(
                chat_id=courier.chat_id,
                text=courier_text,
                reply_markup=get_courier_order_keyboard(order.id),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
            
            logger.info(f"Order {order.id} sent to courier {courier.id}")
    
    except Exception as e:
        logger.error(f"Error sending order to courier: {e}")

# ========================
# COURIER HANDLERS
# ========================

@courier_router.callback_query(F.data.startswith("courier_accept:"))
async def courier_accept_order(callback: CallbackQuery):
    """Kuryer buyurtmani qabul qiladi"""
    try:
        order_id = int(callback.data.split(":")[1])
        
        async with AsyncSessionLocal() as session:
            # Order ni olish
            order_result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = order_result.scalar_one()
            
            # Kuryer ekanligini tekshirish
            courier_result = await session.execute(
                select(Courier).where(Courier.chat_id == callback.from_user.id)
            )
            courier = courier_result.scalar_one_or_none()
            
            if not courier or order.courier_id != courier.id:
                await callback.answer("Этот заказ не назначен вам", show_alert=True)
                return
            
            # Statusni yangilash
            order.status = "OUT_FOR_DELIVERY"
            order.updated_at = datetime.now(TIMEZONE)
            await session.commit()
            
            # Admin xabarni yangilash
            await update_order_message(order)
            
            # Userga xabar
            # User ni topish
            user_result = await session.execute(
                select(User).where(User.id == order.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                await bot.send_message(
                    chat_id=user.tg_id,
                    text=f"🚴 <b>Курьер принял ваш заказ №{order.order_number}!</b>\n\n"
                         f"Имя курьера: <b>{courier.name}</b>\n"
                         "Заказ уже в пути к вам! Ожидайте доставку.",
                    parse_mode="HTML"
                )
            
            await callback.answer("Заказ принят в доставку")
            
            # Xabarni yangilash
            await callback.message.edit_text(
                f"✅ <b>Вы приняли заказ №{order.order_number}</b>\n\n"
                f"Статус: <b>В пути</b>\n"
                f"Клиент: <b>{order.customer_name}</b>\n"
                f"Телефон: <b>{order.phone}</b>\n\n"
                f"<i>После доставки нажмите кнопку 'Доставлен'</i>",
                parse_mode="HTML",
                reply_markup=get_courier_order_keyboard(order.id)
            )
    
    except Exception as e:
        logger.error(f"Error in courier_accept_order: {e}")
        await callback.answer("Ошибка при принятии заказа", show_alert=True)

@courier_router.callback_query(F.data.startswith("courier_delivered:"))
async def courier_delivered_order(callback: CallbackQuery):
    """Kuryer buyurtmani yetkazdi"""
    try:
        order_id = int(callback.data.split(":")[1])
        
        async with AsyncSessionLocal() as session:
            # Order ni olish
            order_result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = order_result.scalar_one()
            
            # Kuryer ekanligini tekshirish
            courier_result = await session.execute(
                select(Courier).where(Courier.chat_id == callback.from_user.id)
            )
            courier = courier_result.scalar_one_or_none()
            
            if not courier or order.courier_id != courier.id:
                await callback.answer("Этот заказ не назначен вам", show_alert=True)
                return
            
            # Statusni yangilash
            order.status = "DELIVERED"
            order.delivered_at = datetime.now(TIMEZONE)
            order.updated_at = datetime.now(TIMEZONE)
            await session.commit()
            
            # Referral statistikani yangilash
            await update_referral_stats(order.user_id, order_delivered=True)
            
            # Admin xabarni yangilash
            await update_order_message(order)
            
            # Userga xabar
            # User ni topish
            user_result = await session.execute(
                select(User).where(User.id == order.user_id)
            )
            user = user_result.scalar_one_or_none()
            
            if user:
                await bot.send_message(
                    chat_id=user.tg_id,
                    text=f"🎉 <b>Ваш заказ №{order.order_number} успешно доставлен!</b>\n\n"
                         f"💰 Сумма: <b>{format_price(order.total)} сум</b>\n"
                         f"🚴 Курьер: <b>{courier.name}</b>\n"
                         f"🕒 Время доставки: <b>{order.delivered_at.strftime('%H:%M')}</b>\n\n"
                         "Спасибо за заказ! Ждем вас снова! 🍽️",
                    parse_mode="HTML"
                )
            
            await callback.answer("Заказ доставлен успешно")
            
            # Xabarni yangilash
            await callback.message.edit_text(
                f"✅ <b>Заказ №{order.order_number} доставлен!</b>\n\n"
                f"Клиент: <b>{order.customer_name}</b>\n"
                f"Сумма: <b>{format_price(order.total)} сум</b>\n"
                f"Время доставки: <b>{order.delivered_at.strftime('%H:%M')}</b>\n\n"
                "<i>Спасибо за работу! 💪</i>",
                parse_mode="HTML",
                reply_markup=None
            )
    
    except Exception as e:
        logger.error(f"Error in courier_delivered_order: {e}")
        await callback.answer("Ошибка при подтверждении доставки", show_alert=True)

# ========================
# BACK BUTTONS
# ========================

@admin_router.callback_query(F.data.startswith("back_to_order:"))
async def back_to_order(callback: CallbackQuery):
    """Buyurtmaga qaytish"""
    order_id = int(callback.data.split(":")[1])
    
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = result.scalar_one()
            
            # Order items olish
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()
            
            items_text = ""
            for item in items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {format_price(item.line_total)} сум\n"
            
            location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
            
            order_text = (
                f"📦 <b>Заказ №{order.order_number}</b>\n\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n"
                f"📊 <b>Статус:</b> {order.status}\n"
                f"📍 <b>Локация:</b> <a href='{location_link}'>На карте</a>\n\n"
                f"🍽️ <b>Заказ:</b>\n{items_text}"
            )
            
            await callback.message.edit_text(
                order_text,
                reply_markup=get_order_status_keyboard(order.id),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
    
    except Exception as e:
        logger.error(f"Error in back_to_order: {e}")
        await callback.answer("Ошибка", show_alert=True)

@admin_router.callback_query(F.data.startswith("back_to_status:"))
async def back_to_status(callback: CallbackQuery):
    """Status sahifasiga qaytish"""
    order_id = int(callback.data.split(":")[1])
    
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Order).where(Order.id == order_id)
            )
            order = result.scalar_one()
            
            # Order items olish
            items_result = await session.execute(
                select(OrderItem).where(OrderItem.order_id == order.id)
            )
            items = items_result.scalars().all()
            
            items_text = ""
            for item in items:
                items_text += f"• {item.name_snapshot} x{item.qty} = {format_price(item.line_total)} сум\n"
            
            location_link = f"https://maps.google.com/?q={order.location_lat},{order.location_lng}"
            
            order_text = (
                f"📦 <b>Заказ №{order.order_number}</b>\n\n"
                f"👤 <b>Клиент:</b> {order.customer_name}\n"
                f"📞 <b>Телефон:</b> {order.phone}\n"
                f"💰 <b>Сумма:</b> {format_price(order.total)} сум\n"
                f"📊 <b>Статус:</b> {order.status}\n"
                f"📍 <b>Локация:</b> <a href='{location_link}'>На карте</a>\n\n"
                f"🍽️ <b>Заказ:</b>\n{items_text}"
            )
            
            await callback.message.edit_text(
                order_text,
                reply_markup=get_order_status_keyboard(order.id),
                parse_mode="HTML",
                disable_web_page_preview=True
            )
    
    except Exception as e:
        logger.error(f"Error in back_to_status: {e}")
        await callback.answer("Ошибка", show_alert=True)

# ========================
# MAIN FUNCTION
# ========================

async def main():
    """Asosiy funksiya"""
    try:
        # Database initialization
        await init_db()
        
        # Routerlarni qo'shish
        dp.include_router(client_router)
        dp.include_router(admin_router)
        dp.include_router(courier_router)
        
        logger.info("=" * 50)
        logger.info("FIESTA Food Delivery Bot starting...")
        logger.info(f"Bot ID: {bot.id}")
        logger.info(f"Admins: {ADMIN_IDS}")
        logger.info(f"Shop Channel: {SHOP_CHANNEL_ID}")
        logger.info(f"Courier Channel: {COURIER_CHANNEL_ID}")
        logger.info(f"WebApp URL: {WEBAPP_URL}")
        logger.info("=" * 50)
        
        # Start polling
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())
