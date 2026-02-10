import os
import json
import hmac
import uuid
import hashlib
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Update,
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    WebAppInfo,
)
from aiogram.fsm.storage.redis import RedisStorage

from sqlalchemy import (
    String, Integer, BigInteger, Boolean, DateTime, ForeignKey, Text, Float,
    UniqueConstraint, select, func, desc, asc, update
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fiesta")


# =========================
# CONFIG (ENV)
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DB_URL = os.getenv("DB_URL", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()

WEBAPP_URL = os.getenv("WEBAPP_URL", "").strip()

# Public base URL of this backend (Railway domain), required for webhook
API_PUBLIC_BASE = os.getenv("API_PUBLIC_BASE", "").strip()

# Optional channels (can be overridden/saved in DB settings via admin panel)
SHOP_CHANNEL_ID_ENV = os.getenv("SHOP_CHANNEL_ID", "").strip()
COURIER_CHANNEL_ID_ENV = os.getenv("COURIER_CHANNEL_ID", "").strip()

ADMIN_IDS_RAW = os.getenv("ADMIN_IDS", "").strip()  # "1,2,3" or "1 2 3"
ADMIN_IDS: List[int] = []
if ADMIN_IDS_RAW:
    parts = [p.strip() for p in ADMIN_IDS_RAW.replace(",", " ").split()]
    ADMIN_IDS = [int(p) for p in parts if p.isdigit()]

PORT = int(os.getenv("PORT", "8080"))

if not BOT_TOKEN:
    raise RuntimeError("ENV BOT_TOKEN is required")
if not DB_URL:
    raise RuntimeError("ENV DB_URL is required")
if not REDIS_URL:
    raise RuntimeError("ENV REDIS_URL is required")
if not WEBAPP_URL:
    raise RuntimeError("ENV WEBAPP_URL is required")
if not API_PUBLIC_BASE:
    raise RuntimeError("ENV API_PUBLIC_BASE is required (public URL for webhook)")
if not ADMIN_IDS:
    raise RuntimeError("ENV ADMIN_IDS is required (at least 1 admin)")

API_PUBLIC_BASE = API_PUBLIC_BASE.rstrip("/")
WEBAPP_URL = WEBAPP_URL.rstrip("/")


# =========================
# DB
# =========================
engine = create_async_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# =========================
# MODELS
# =========================
class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), nullable=False)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    full_name: Mapped[str] = mapped_column(String(255))
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ref_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    promo_given_15: Mapped[bool] = mapped_column(Boolean, default=False)

    referrals: Mapped[List["User"]] = relationship(remote_side="User.id")


class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Food(Base):
    __tablename__ = "foods"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))
    name: Mapped[str] = mapped_column(String(150), index=True)
    description: Mapped[str] = mapped_column(String(255), default="")
    price: Mapped[int] = mapped_column(Integer)  # sum
    rating: Mapped[float] = mapped_column(Float, default=4.5)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Promo(Base):
    __tablename__ = "promos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    discount_percent: Mapped[int] = mapped_column(Integer)  # 1..90
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_limit: Mapped[int] = mapped_column(Integer, default=999999)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Courier(Base):
    __tablename__ = "couriers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    order_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    customer_name: Mapped[str] = mapped_column(String(255))
    phone: Mapped[str] = mapped_column(String(40))
    comment: Mapped[str] = mapped_column(String(500), default="")

    total: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="NEW")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    location_lat: Mapped[float] = mapped_column(Float)
    location_lng: Mapped[float] = mapped_column(Float)

    courier_id: Mapped[Optional[int]] = mapped_column(ForeignKey("couriers.id"), nullable=True)

    admin_channel_chat_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    admin_channel_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    promo_code_used: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    items: Mapped[List["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    food_id: Mapped[int] = mapped_column(Integer)

    name_snapshot: Mapped[str] = mapped_column(String(200))
    price_snapshot: Mapped[int] = mapped_column(Integer)
    qty: Mapped[int] = mapped_column(Integer)
    line_total: Mapped[int] = mapped_column(Integer)

    order: Mapped["Order"] = relationship(back_populates="items")


# =========================
# STATUS
# =========================
STATUS_LABEL = {
    "NEW": "Принят",
    "CONFIRMED": "Подтвержден",
    "COOKING": "Готовится",
    "COURIER_ASSIGNED": "Курьер назначен",
    "OUT_FOR_DELIVERY": "Передан курьеру",
    "DELIVERED": "Доставлен",
    "CANCELED": "Отменен",
}
ACTIVE_STATUSES = {"NEW", "CONFIRMED", "COOKING", "COURIER_ASSIGNED", "OUT_FOR_DELIVERY"}


# =========================
# TELEGRAM / DISPATCHER (WEBHOOK)
# =========================
storage = RedisStorage.from_url(REDIS_URL)
dp = Dispatcher(storage=storage)
bot = Bot(BOT_TOKEN)

router_client = Router()
router_admin = Router()
router_courier = Router()

dp.include_router(router_client)
dp.include_router(router_admin)
dp.include_router(router_courier)


# =========================
# KEYBOARDS
# =========================
def kb_client_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")],
        ],
        resize_keyboard=True
    )


def kb_shop_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🛍 Заказать", url=WEBAPP_URL)]
    ])


def kb_admin_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🍔 Taomlar", callback_data="admin:foods")],
        [InlineKeyboardButton(text="📂 Kategoriyalar", callback_data="admin:categories")],
        [InlineKeyboardButton(text="🎁 Promokodlar", callback_data="admin:promos")],
        [InlineKeyboardButton(text="📊 Statistika", callback_data="admin:stats")],
        [InlineKeyboardButton(text="🚴 Kuryerlar", callback_data="admin:couriers")],
        [InlineKeyboardButton(text="📦 Aktiv buyurtmalar", callback_data="admin:active_orders")],
        [InlineKeyboardButton(text="⚙️ Sozlamalar", callback_data="admin:settings")],
    ])


def kb_admin_order_status(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Подтвержден", callback_data=f"order:set:{order_id}:CONFIRMED"),
            InlineKeyboardButton(text="🍳 Готовится", callback_data=f"order:set:{order_id}:COOKING"),
        ],
        [
            InlineKeyboardButton(text="🚴 Курьер", callback_data=f"order:courier:{order_id}")
        ],
        [
            InlineKeyboardButton(text="❌ Отменен", callback_data=f"order:set:{order_id}:CANCELED")
        ]
    ])


def kb_courier_actions(order_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="✅ Qabul qildim", callback_data=f"courier:accept:{order_id}"),
            InlineKeyboardButton(text="📦 Yetkazildi", callback_data=f"courier:delivered:{order_id}")
        ]
    ])


def kb_pick_courier(order_id: int, couriers: List[Courier]) -> InlineKeyboardMarkup:
    rows = []
    for c in couriers:
        rows.append([InlineKeyboardButton(text=f"🚴 {c.name}", callback_data=f"courier:assign:{order_id}:{c.id}")])
    rows.append([InlineKeyboardButton(text="⬅️ Orqaga", callback_data=f"order:back:{order_id}")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# =========================
# INITDATA VERIFY (Telegram WebApp)
# =========================
def verify_telegram_init_data(init_data: str, bot_token: str) -> Dict[str, Any]:
    if not init_data:
        raise ValueError("Empty initData")

    pairs = [p.split("=", 1) for p in init_data.split("&") if "=" in p]
    data = {k: v for k, v in pairs}
    received_hash = data.pop("hash", None)
    if not received_hash:
        raise ValueError("Missing hash in initData")

    secret = hashlib.sha256(bot_token.encode()).digest()
    check_string = "\n".join(f"{k}={v}" for k, v in sorted(data.items()))
    computed = hmac.new(secret, check_string.encode(), hashlib.sha256).hexdigest()
    if computed != received_hash:
        raise ValueError("Invalid initData hash")

    user_raw = data.get("user")
    if not user_raw:
        raise ValueError("Missing user in initData")
    return json.loads(user_raw)


# =========================
# DB HELPERS
# =========================
async def get_setting(session: AsyncSession, key: str) -> Optional[str]:
    row = await session.get(Setting, key)
    return row.value if row else None


async def set_setting(session: AsyncSession, key: str, value: str) -> None:
    row = await session.get(Setting, key)
    if row:
        row.value = value
    else:
        session.add(Setting(key=key, value=value))


async def get_channels(session: AsyncSession) -> Tuple[Optional[int], Optional[int]]:
    # Priority: DB setting -> ENV -> None
    shop = await get_setting(session, "SHOP_CHANNEL_ID")
    courier = await get_setting(session, "COURIER_CHANNEL_ID")
    shop_id = int(shop) if shop else (int(SHOP_CHANNEL_ID_ENV) if SHOP_CHANNEL_ID_ENV else None)
    courier_id = int(courier) if courier else (int(COURIER_CHANNEL_ID_ENV) if COURIER_CHANNEL_ID_ENV else None)
    return shop_id, courier_id


async def upsert_user(session: AsyncSession, tg_id: int, username: Optional[str], full_name: str,
                     ref_by_user_id: Optional[int]) -> User:
    q = await session.execute(select(User).where(User.tg_id == tg_id))
    user = q.scalar_one_or_none()
    if user:
        user.username = username
        user.full_name = full_name
        # set referral only once
        if ref_by_user_id and user.ref_by_user_id is None and ref_by_user_id != user.id:
            user.ref_by_user_id = ref_by_user_id
        return user

    user = User(
        tg_id=tg_id,
        username=username,
        full_name=full_name,
        joined_at=utcnow(),
        ref_by_user_id=ref_by_user_id,
    )
    session.add(user)
    await session.flush()  # get id
    return user


async def seed_demo_data(session: AsyncSession) -> None:
    # categories fixed list
    cat_names = ["All", "Lavash", "Burger", "Xaggi", "Shaurma", "Hotdog", "Combo", "Sneki", "Sous", "Napitki"]
    existing = (await session.execute(select(func.count(Category.id)))).scalar_one()
    if existing and existing > 0:
        return

    cats = []
    for n in cat_names[1:]:  # skip "All" (virtual in UI)
        c = Category(name=n, is_active=True)
        session.add(c)
        cats.append(c)
    await session.flush()

    # 3 foods per category demo
    for c in cats:
        for i in range(1, 4):
            session.add(Food(
                category_id=c.id,
                name=f"{c.name} #{i}",
                description=f"Demo {c.name} taomi #{i}",
                price=35000 + i * 5000,
                rating=4.2 + i * 0.2,
                is_new=(i == 3),
                is_active=True,
                image_url=None,
                created_at=utcnow()
            ))


# =========================
# TELEGRAM NOTIFY
# =========================
def format_order_items(items: List[OrderItem]) -> str:
    lines = []
    for it in items:
        lines.append(f"• {it.name_snapshot} x{it.qty} = {it.line_total} сум")
    return "\n".join(lines)


def maps_link(lat: float, lng: float) -> str:
    return f"https://maps.google.com/?q={lat},{lng}"


async def notify_user(order: Order) -> None:
    try:
        text = (
            f"Ваш заказ принят ✅\n"
            f"🆔 Заказ №{order.order_number}\n"
            f"💰 Сумма: {order.total} сум\n"
            f"📦 Статус: {STATUS_LABEL.get(order.status, order.status)}"
        )
        await bot.send_message(order.user_id_to_tg, text)  # patched below
    except Exception as e:
        log.warning(f"notify_user failed: {e}")


async def send_or_edit_admin_post(session: AsyncSession, order: Order, user: User) -> None:
    shop_channel_id, _ = await get_channels(session)
    if not shop_channel_id:
        return

    item_text = format_order_items(order.items)
    text = (
        f"🆕 Новый заказ №{order.order_number}\n"
        f"👤 Пользователь: {user.full_name} (@{user.username or '—'})\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total}\n"
        f"🕒 Время: {order.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"📍 Локация: {order.location_lat},{order.location_lng}\n"
        f"🔗 Карта: {maps_link(order.location_lat, order.location_lng)}\n"
        f"📦 Статус: {STATUS_LABEL.get(order.status, order.status)}\n\n"
        f"🍽️ Заказ:\n{item_text}"
    )

    if order.admin_channel_chat_id and order.admin_channel_message_id:
        try:
            await bot.edit_message_text(
                chat_id=order.admin_channel_chat_id,
                message_id=order.admin_channel_message_id,
                text=text,
                reply_markup=None if order.status == "DELIVERED" else kb_admin_order_status(order.id),
                disable_web_page_preview=True,
            )
            return
        except Exception:
            pass

    try:
        msg = await bot.send_message(
            chat_id=shop_channel_id,
            text=text,
            reply_markup=kb_admin_order_status(order.id),
            disable_web_page_preview=True,
        )
        order.admin_channel_chat_id = shop_channel_id
        order.admin_channel_message_id = msg.message_id
    except Exception as e:
        log.warning(f"send admin post failed: {e}")


async def send_order_to_courier(session: AsyncSession, order: Order, courier: Courier, user: User) -> None:
    _, courier_channel_id = await get_channels(session)

    item_text = format_order_items(order.items)
    text = (
        f"🚴 Новый заказ №{order.order_number}\n"
        f"👤 Клиент: {order.customer_name}\n"
        f"📞 Телефон: {order.phone}\n"
        f"💰 Сумма: {order.total}\n"
        f"📍 Локация: {maps_link(order.location_lat, order.location_lng)}\n\n"
        f"🍽️ Список:\n{item_text}"
    )

    target_chat = courier_channel_id or courier.chat_id
    await bot.send_message(
        chat_id=target_chat,
        text=text,
        reply_markup=kb_courier_actions(order.id),
        disable_web_page_preview=True,
    )

    # also ping courier directly if channel used
    if courier_channel_id and courier.chat_id != courier_channel_id:
        try:
            await bot.send_message(
                chat_id=courier.chat_id,
                text=f"Вам назначен заказ №{order.order_number}. Проверьте канал/сообщение.",
            )
        except Exception:
            pass


# =========================
# IMPORTANT PATCH:
# We store order.user_id as DB FK to users.id, but telegram needs tg_id.
# We'll provide helper for tg_id
# =========================
async def get_user_tg_id(session: AsyncSession, user_id: int) -> int:
    row = await session.execute(select(User.tg_id).where(User.id == user_id))
    tg_id = row.scalar_one()
    return int(tg_id)


# =========================
# CLIENT HANDLERS
# =========================
@router_client.message(CommandStart())
async def cmd_start(message: Message, command: CommandStart):
    ref = None
    if command.args:
        try:
            ref = int(command.args.strip())
        except Exception:
            ref = None

    async with SessionLocal() as session:
        ref_by_user_id = None
        if ref:
            # ref is user's DB id
            ref_user = await session.get(User, ref)
            if ref_user:
                ref_by_user_id = ref_user.id

        user = await upsert_user(
            session=session,
            tg_id=message.from_user.id,
            username=message.from_user.username,
            full_name=message.from_user.full_name or "User",
            ref_by_user_id=ref_by_user_id,
        )

        await seed_demo_data(session)
        await session.commit()

        text = (
            f"Добро пожаловать в FIESTA! {user.full_name}\n"
            f"Для заказа перейдите по кнопке ➡️\n"
            f"🛍 Заказать"
        )
        await message.answer(text, reply_markup=kb_client_menu())


@router_client.message(Command("shop"))
async def cmd_shop(message: Message):
    await message.answer("Чтобы открыть наш магазин, нажмите кнопку ниже", reply_markup=kb_shop_inline())


@router_client.message(F.text == "ℹ️ Информация о нас")
async def about_us(message: Message):
    await message.answer(
        "🌟 Добро Пожаловать в FIESTA !\n"
        "📍 Наш адрес:Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢﻿ Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙﻿ Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)"
    )


@router_client.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message):
    async with SessionLocal() as session:
        user_row = await session.execute(select(User).where(User.tg_id == message.from_user.id))
        user = user_row.scalar_one_or_none()
        if not user:
            await message.answer("Пожалуйста нажмите /start")
            return

        q = await session.execute(
            select(Order).where(Order.user_id == user.id).order_by(desc(Order.created_at)).limit(10)
        )
        orders = q.scalars().all()
        if not orders:
            await message.answer(
                "В данный момент у вас нет активных заказов в нашем магазине.\n"
                "Чтобы открыть магазин, введите команду — /shop"
            )
            return

        # load items for each order
        text_lines = []
        for o in orders:
            items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == o.id))
            items = items_q.scalars().all()
            items_txt = "\n".join([f"   - {it.name_snapshot} x{it.qty} = {it.line_total} сум" for it in items]) or "   - —"
            text_lines.append(
                f"🆔 Заказ №{o.order_number} | {o.created_at.astimezone().strftime('%Y-%m-%d %H:%M')} | "
                f"💰 {o.total} | 📦 {STATUS_LABEL.get(o.status, o.status)}\n{items_txt}"
            )
        await message.answer("\n\n".join(text_lines))


@router_client.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message):
    async with SessionLocal() as session:
        user_row = await session.execute(select(User).where(User.tg_id == message.from_user.id))
        user = user_row.scalar_one_or_none()
        if not user:
            await message.answer("Пожалуйста нажмите /start")
            return

        # referral stats
        ref_count = (await session.execute(select(func.count(User.id)).where(User.ref_by_user_id == user.id))).scalar_one()
        orders_count = (await session.execute(select(func.count(Order.id)).where(Order.user_id == user.id))).scalar_one()
        delivered_count = (await session.execute(
            select(func.count(Order.id)).where(Order.user_id == user.id, Order.status == "DELIVERED")
        )).scalar_one()

        me = await bot.get_me()
        link = f"https://t.me/{me.username}?start={user.id}"

        text = (
            "За приглашение друга, вы можете получить промо-код от нас\n"
            f"👥 Вы пригласили {ref_count} человек\n"
            f"🛒 Оформили заказов: {orders_count}\n"
            f"💰 Оплатили заказов: {delivered_count}\n"
            f"👤 Ваша реферальная ссылка: {link}\n"
            "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
        )

        # auto promo for >=3 referrals once
        if ref_count >= 3 and not user.promo_given_15:
            code = f"REF15-{user.id}-{uuid.uuid4().hex[:4].upper()}"
            promo = Promo(code=code, discount_percent=15, expires_at=None, usage_limit=1, used_count=0, is_active=True)
            session.add(promo)
            user.promo_given_15 = True
            await session.commit()
            await message.answer(text + f"\n\n🎁 Ваш промо-код: {code} (скидка 15%)")
            return

        await message.answer(text)


# =========================
# WEBAPP -> BOT (order_create)
# =========================
@router_client.message(F.web_app_data)
async def webapp_data(message: Message):
    try:
        payload = json.loads(message.web_app_data.data)
    except Exception:
        await message.answer("Ошибка данных WebApp.")
        return

    if payload.get("type") != "order_create":
        await message.answer("Неизвестный тип данных.")
        return

    items = payload.get("items") or []
    total = int(payload.get("total") or 0)
    if total < 50000:
        await message.answer("Минимальная сумма заказа: 50 000 сум.")
        return

    customer_name = (payload.get("customer_name") or message.from_user.full_name or "User").strip()
    phone = (payload.get("phone") or "").strip()
    comment = (payload.get("comment") or "").strip()
    location = payload.get("location") or {}
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        await message.answer("Локация обязательна (lat/lng).")
        return

    promo_code = (payload.get("promo_code") or "").strip().upper() or None

    async with SessionLocal() as session:
        user_row = await session.execute(select(User).where(User.tg_id == message.from_user.id))
        user = user_row.scalar_one_or_none()
        if not user:
            await message.answer("Пожалуйста нажмите /start и попробуйте снова.")
            return

        # promo validate (optional)
        discount_percent = 0
        if promo_code:
            pr = await session.execute(select(Promo).where(Promo.code == promo_code))
            promo = pr.scalar_one_or_none()
            if not promo or not promo.is_active:
                await message.answer("Промокод недействителен.")
                return
            if promo.expires_at and promo.expires_at < utcnow():
                await message.answer("Промокод истёк.")
                return
            if promo.used_count >= promo.usage_limit:
                await message.answer("Лимит промокода исчерпан.")
                return
            discount_percent = int(promo.discount_percent)

        # compute total from items safely (server-side)
        # Items from UI include price; but we verify by DB price (best-effort).
        food_ids = [int(x.get("food_id")) for x in items if str(x.get("food_id", "")).isdigit()]
        foods = {}
        if food_ids:
            fq = await session.execute(select(Food).where(Food.id.in_(food_ids)))
            for f in fq.scalars().all():
                foods[f.id] = f

        computed_total = 0
        order_items: List[OrderItem] = []
        for x in items:
            fid = int(x.get("food_id"))
            qty = int(x.get("qty") or 0)
            if qty <= 0:
                continue
            f = foods.get(fid)
            if not f or not f.is_active:
                continue
            line_total = int(f.price) * qty
            computed_total += line_total
            order_items.append(OrderItem(
                food_id=fid,
                name_snapshot=f.name,
                price_snapshot=int(f.price),
                qty=qty,
                line_total=line_total,
            ))

        if computed_total < 50000 or not order_items:
            await message.answer("Корзина пуста или сумма меньше 50 000 сум.")
            return

        final_total = computed_total
        if discount_percent > 0:
            final_total = max(0, int(round(computed_total * (100 - discount_percent) / 100)))
            # mark promo used
            promo = (await session.execute(select(Promo).where(Promo.code == promo_code))).scalar_one()
            promo.used_count += 1

        order_number = uuid.uuid4().hex[:10].upper()
        order = Order(
            order_number=order_number,
            user_id=user.id,
            customer_name=customer_name,
            phone=phone,
            comment=comment,
            total=final_total,
            status="NEW",
            created_at=utcnow(),
            updated_at=utcnow(),
            location_lat=float(lat),
            location_lng=float(lng),
            courier_id=None,
            promo_code_used=promo_code,
        )
        session.add(order)
        await session.flush()
        for it in order_items:
            it.order_id = order.id
            session.add(it)

        await session.commit()

        # user notify
        await message.answer(
            f"Ваш заказ принят ✅\n"
            f"🆔 Заказ №{order.order_number}\n"
            f"💰 Сумма: {order.total} сум\n"
            f"📦 Статус: Принят",
            reply_markup=kb_client_menu()
        )

        # admin channel post
        # reload items relationship
        items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == order.id))
        order.items = items_q.scalars().all()

        # attach tg_id for user messaging in updates
        # (we edit later by helper)
        order.user_id_to_tg = message.from_user.id  # runtime only

        await send_or_edit_admin_post(session, order, user)
        await session.commit()


# =========================
# ADMIN PANEL
# =========================
def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


@router_admin.message(Command("admin"))
async def admin_panel(message: Message):
    if not is_admin(message.from_user.id):
        return
    await message.answer("Админ панель:", reply_markup=kb_admin_menu())


@router_admin.callback_query(F.data.startswith("admin:"))
async def admin_menu_click(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("Нет доступа", show_alert=True)
        return

    section = call.data.split(":", 1)[1]

    if section == "settings":
        async with SessionLocal() as session:
            shop, courier = await get_channels(session)
        await call.message.edit_text(
            f"⚙️ Sozlamalar\n"
            f"SHOP_CHANNEL_ID: {shop or '—'}\n"
            f"COURIER_CHANNEL_ID: {courier or '—'}\n\n"
            f"Yozing:\n"
            f"`/set_shop -100...`\n"
            f"`/set_courier -100...`",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    if section == "categories":
        await call.message.edit_text(
            "📂 Kategoriyalar\n"
            "Qo‘shish: `/add_category Nomi`\n"
            "O‘chirish: `/del_category Nomi`\n"
            "Ro‘yxat: `/list_categories`",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    if section == "foods":
        await call.message.edit_text(
            "🍔 Taomlar\n"
            "Qo‘shish: `/add_food <category>;<name>;<price>;<rating>;<is_new 0/1>;<image_url(optional)>;<desc(optional)`\n"
            "Misol: `/add_food Lavash;Lavash #1;45000;4.7;1;;Zo'r lavash`\n"
            "O‘chirish: `/del_food <id>`\n"
            "Ro‘yxat: `/list_foods <category(optional)>`",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    if section == "couriers":
        await call.message.edit_text(
            "🚴 Kuryerlar\n"
            "Qo‘shish: `/add_courier <chat_id>;<name>`\n"
            "O‘chirish(disable): `/disable_courier <chat_id>`\n"
            "Ro‘yxat: `/list_couriers`",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    if section == "active_orders":
        async with SessionLocal() as session:
            q = await session.execute(select(Order).where(Order.status.in_(list(ACTIVE_STATUSES))).order_by(desc(Order.created_at)).limit(20))
            orders = q.scalars().all()
        if not orders:
            await call.message.edit_text("📦 Aktiv buyurtmalar yo‘q.", reply_markup=kb_admin_menu())
            await call.answer()
            return

        lines = ["📦 Aktiv buyurtmalar:"]
        for o in orders:
            lines.append(f"• №{o.order_number} | {o.total} | {STATUS_LABEL.get(o.status, o.status)} | id={o.id}")
        await call.message.edit_text("\n".join(lines), reply_markup=kb_admin_menu())
        await call.answer()
        return

    if section == "stats":
        async with SessionLocal() as session:
            today = datetime.now().date()
            # counts
            orders_today = (await session.execute(select(func.count(Order.id)).where(func.date(Order.created_at) == today))).scalar_one()
            delivered_today = (await session.execute(select(func.count(Order.id)).where(func.date(Order.created_at) == today, Order.status == "DELIVERED"))).scalar_one()
            revenue_today = (await session.execute(select(func.coalesce(func.sum(Order.total), 0)).where(func.date(Order.created_at) == today, Order.status == "DELIVERED"))).scalar_one()
            active = (await session.execute(select(func.count(Order.id)).where(Order.status.in_(list(ACTIVE_STATUSES))))).scalar_one()

        await call.message.edit_text(
            "📊 Statistika (Bugun)\n"
            f"🛒 Buyurtmalar: {orders_today}\n"
            f"✅ Delivered: {delivered_today}\n"
            f"💰 Tushum: {int(revenue_today)}\n"
            f"📦 Active: {active}",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    if section == "promos":
        await call.message.edit_text(
            "🎁 Promokodlar\n"
            "Yaratish: `/add_promo CODE;PERCENT(1-90);usage_limit;expires_at(YYYY-MM-DD or empty)`\n"
            "Misol: `/add_promo SALE10;10;100;2026-12-31`\n"
            "Ro‘yxat: `/list_promos`\n"
            "Disable: `/disable_promo CODE`",
            reply_markup=kb_admin_menu()
        )
        await call.answer()
        return

    await call.answer()


@router_admin.message(Command("set_shop"))
async def set_shop(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Misol: /set_shop -1003530497437")
        return
    val = parts[1].strip()
    async with SessionLocal() as session:
        await set_setting(session, "SHOP_CHANNEL_ID", val)
        await session.commit()
    await message.answer(f"✅ SHOP_CHANNEL_ID set: {val}")


@router_admin.message(Command("set_courier"))
async def set_courier(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Misol: /set_courier -1003707946746")
        return
    val = parts[1].strip()
    async with SessionLocal() as session:
        await set_setting(session, "COURIER_CHANNEL_ID", val)
        await session.commit()
    await message.answer(f"✅ COURIER_CHANNEL_ID set: {val}")


@router_admin.message(Command("add_category"))
async def add_category(message: Message):
    if not is_admin(message.from_user.id):
        return
    name = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if not name:
        await message.answer("Misol: /add_category Lavash")
        return
    async with SessionLocal() as session:
        session.add(Category(name=name, is_active=True))
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            await message.answer("Bu kategoriya bor.")
            return
    await message.answer("✅ Qo‘shildi")


@router_admin.message(Command("del_category"))
async def del_category(message: Message):
    if not is_admin(message.from_user.id):
        return
    name = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if not name:
        await message.answer("Misol: /del_category Lavash")
        return
    async with SessionLocal() as session:
        q = await session.execute(select(Category).where(Category.name == name))
        c = q.scalar_one_or_none()
        if not c:
            await message.answer("Topilmadi")
            return
        c.is_active = False
        await session.commit()
    await message.answer("✅ Disabled")


@router_admin.message(Command("list_categories"))
async def list_categories(message: Message):
    if not is_admin(message.from_user.id):
        return
    async with SessionLocal() as session:
        q = await session.execute(select(Category).order_by(asc(Category.name)))
        cats = q.scalars().all()
    lines = ["📂 Kategoriyalar:"]
    for c in cats:
        lines.append(f"• {c.name} | active={c.is_active} | id={c.id}")
    await message.answer("\n".join(lines))


@router_admin.message(Command("add_food"))
async def add_food(message: Message):
    if not is_admin(message.from_user.id):
        return
    raw = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if not raw or ";" not in raw:
        await message.answer("Format: /add_food Category;Name;Price;Rating;is_new(0/1);image_url(optional);desc(optional)")
        return
    parts = [p.strip() for p in raw.split(";")]
    while len(parts) < 7:
        parts.append("")
    cat_name, name, price_s, rating_s, is_new_s, image_url, desc_ = parts[:7]
    try:
        price = int(price_s)
        rating = float(rating_s)
        is_new = bool(int(is_new_s))
    except Exception:
        await message.answer("Price/rating/is_new noto‘g‘ri.")
        return

    async with SessionLocal() as session:
        cq = await session.execute(select(Category).where(Category.name == cat_name))
        cat = cq.scalar_one_or_none()
        if not cat:
            await message.answer("Kategoriya topilmadi.")
            return
        f = Food(
            category_id=cat.id,
            name=name,
            description=desc_ or "",
            price=price,
            rating=rating,
            is_new=is_new,
            is_active=True,
            image_url=image_url or None,
            created_at=utcnow()
        )
        session.add(f)
        await session.commit()
    await message.answer("✅ Taom qo‘shildi")


@router_admin.message(Command("del_food"))
async def del_food(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].isdigit():
        await message.answer("Misol: /del_food 12")
        return
    fid = int(parts[1])
    async with SessionLocal() as session:
        f = await session.get(Food, fid)
        if not f:
            await message.answer("Topilmadi")
            return
        f.is_active = False
        await session.commit()
    await message.answer("✅ Disabled")


@router_admin.message(Command("list_foods"))
async def list_foods(message: Message):
    if not is_admin(message.from_user.id):
        return
    cat = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    async with SessionLocal() as session:
        if cat:
            cq = await session.execute(select(Category).where(Category.name == cat))
            c = cq.scalar_one_or_none()
            if not c:
                await message.answer("Kategoriya topilmadi")
                return
            q = await session.execute(select(Food).where(Food.category_id == c.id).order_by(desc(Food.created_at)).limit(50))
        else:
            q = await session.execute(select(Food).order_by(desc(Food.created_at)).limit(50))
        foods = q.scalars().all()
    lines = ["🍔 Taomlar:"]
    for f in foods:
        lines.append(f"• id={f.id} | {f.name} | {f.price} | rating={f.rating} | new={f.is_new} | active={f.is_active}")
    await message.answer("\n".join(lines))


@router_admin.message(Command("add_courier"))
async def add_courier(message: Message):
    if not is_admin(message.from_user.id):
        return
    raw = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if ";" not in raw:
        await message.answer("Format: /add_courier <chat_id>;<name>")
        return
    chat_id_s, name = [p.strip() for p in raw.split(";", 1)]
    try:
        chat_id = int(chat_id_s)
    except Exception:
        await message.answer("chat_id noto‘g‘ri")
        return
    async with SessionLocal() as session:
        c = Courier(chat_id=chat_id, name=name, is_active=True, created_at=utcnow())
        session.add(c)
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            await message.answer("Bu kuryer bor.")
            return
    await message.answer("✅ Kuryer qo‘shildi")


@router_admin.message(Command("disable_courier"))
async def disable_courier(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Misol: /disable_courier 123456")
        return
    try:
        chat_id = int(parts[1])
    except Exception:
        await message.answer("chat_id noto‘g‘ri")
        return
    async with SessionLocal() as session:
        q = await session.execute(select(Courier).where(Courier.chat_id == chat_id))
        c = q.scalar_one_or_none()
        if not c:
            await message.answer("Topilmadi")
            return
        c.is_active = False
        await session.commit()
    await message.answer("✅ Disabled")


@router_admin.message(Command("list_couriers"))
async def list_couriers(message: Message):
    if not is_admin(message.from_user.id):
        return
    async with SessionLocal() as session:
        q = await session.execute(select(Courier).order_by(desc(Courier.created_at)))
        cs = q.scalars().all()
    lines = ["🚴 Kuryerlar:"]
    for c in cs:
        lines.append(f"• {c.name} | chat_id={c.chat_id} | active={c.is_active} | id={c.id}")
    await message.answer("\n".join(lines))


@router_admin.message(Command("add_promo"))
async def add_promo(message: Message):
    if not is_admin(message.from_user.id):
        return
    raw = message.text.split(maxsplit=1)[1].strip() if len(message.text.split(maxsplit=1)) > 1 else ""
    if ";" not in raw:
        await message.answer("Format: /add_promo CODE;PERCENT;usage_limit;expires_at(YYYY-MM-DD or empty)")
        return
    parts = [p.strip() for p in raw.split(";")]
    while len(parts) < 4:
        parts.append("")
    code, percent_s, limit_s, exp_s = parts[:4]
    code = code.upper()
    try:
        percent = int(percent_s)
        limit = int(limit_s)
        if percent < 1 or percent > 90:
            raise ValueError()
    except Exception:
        await message.answer("percent/limit noto‘g‘ri")
        return
    expires = None
    if exp_s:
        try:
            expires = datetime.fromisoformat(exp_s).replace(tzinfo=timezone.utc)
        except Exception:
            await message.answer("expires_at format: YYYY-MM-DD")
            return

    async with SessionLocal() as session:
        session.add(Promo(
            code=code,
            discount_percent=percent,
            expires_at=expires,
            usage_limit=limit,
            used_count=0,
            is_active=True,
            created_at=utcnow()
        ))
        try:
            await session.commit()
        except Exception:
            await session.rollback()
            await message.answer("Bu promo bor.")
            return
    await message.answer("✅ Promo yaratildi")


@router_admin.message(Command("disable_promo"))
async def disable_promo(message: Message):
    if not is_admin(message.from_user.id):
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Misol: /disable_promo SALE10")
        return
    code = parts[1].strip().upper()
    async with SessionLocal() as session:
        q = await session.execute(select(Promo).where(Promo.code == code))
        p = q.scalar_one_or_none()
        if not p:
            await message.answer("Topilmadi")
            return
        p.is_active = False
        await session.commit()
    await message.answer("✅ Disabled")


@router_admin.message(Command("list_promos"))
async def list_promos(message: Message):
    if not is_admin(message.from_user.id):
        return
    async with SessionLocal() as session:
        q = await session.execute(select(Promo).order_by(desc(Promo.created_at)).limit(50))
        ps = q.scalars().all()
    lines = ["🎁 Promokodlar:"]
    for p in ps:
        exp = p.expires_at.date().isoformat() if p.expires_at else "—"
        lines.append(f"• {p.code} | {p.discount_percent}% | used {p.used_count}/{p.usage_limit} | exp={exp} | active={p.is_active}")
    await message.answer("\n".join(lines))


# =========================
# ORDER STATUS CALLBACKS (admin channel)
# =========================
@router_admin.callback_query(F.data.startswith("order:set:"))
async def order_set_status(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("Нет доступа", show_alert=True)
        return
    _, _, order_id_s, new_status = call.data.split(":", 3)
    order_id = int(order_id_s)

    async with SessionLocal() as session:
        o = await session.get(Order, order_id)
        if not o:
            await call.answer("Order not found", show_alert=True)
            return
        o.status = new_status
        o.updated_at = utcnow()
        if new_status == "DELIVERED":
            o.delivered_at = utcnow()

        # load user + items
        user = await session.get(User, o.user_id)
        items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == o.id))
        o.items = items_q.scalars().all()

        # update admin post
        await send_or_edit_admin_post(session, o, user)
        await session.commit()

        # notify user
        tg_id = await get_user_tg_id(session, o.user_id)
        try:
            await bot.send_message(
                tg_id,
                f"Ваш заказ №{o.order_number}\n📦 Статус: {STATUS_LABEL.get(o.status, o.status)}"
            )
        except Exception:
            pass

    await call.answer("✅")


@router_admin.callback_query(F.data.startswith("order:courier:"))
async def order_choose_courier(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("Нет доступа", show_alert=True)
        return
    order_id = int(call.data.split(":")[2])

    async with SessionLocal() as session:
        o = await session.get(Order, order_id)
        if not o:
            await call.answer("Order not found", show_alert=True)
            return
        q = await session.execute(select(Courier).where(Courier.is_active == True).order_by(asc(Courier.name)))
        couriers = q.scalars().all()
        if not couriers:
            await call.answer("Kuryer yo‘q. /add_courier bilan qo‘shing", show_alert=True)
            return

    await call.message.edit_reply_markup(reply_markup=kb_pick_courier(order_id, couriers))
    await call.answer()


@router_admin.callback_query(F.data.startswith("order:back:"))
async def order_back(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("Нет доступа", show_alert=True)
        return
    order_id = int(call.data.split(":")[2])
    await call.message.edit_reply_markup(reply_markup=kb_admin_order_status(order_id))
    await call.answer()


@router_admin.callback_query(F.data.startswith("courier:assign:"))
async def assign_courier(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("Нет доступа", show_alert=True)
        return
    _, _, order_id_s, courier_id_s = call.data.split(":")
    order_id = int(order_id_s)
    courier_id = int(courier_id_s)

    async with SessionLocal() as session:
        o = await session.get(Order, order_id)
        c = await session.get(Courier, courier_id)
        if not o or not c or not c.is_active:
            await call.answer("Not found", show_alert=True)
            return

        o.courier_id = c.id
        o.status = "COURIER_ASSIGNED"
        o.updated_at = utcnow()

        user = await session.get(User, o.user_id)
        items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == o.id))
        o.items = items_q.scalars().all()

        await send_or_edit_admin_post(session, o, user)
        await send_order_to_courier(session, o, c, user)

        await session.commit()

        # notify user
        tg_id = await get_user_tg_id(session, o.user_id)
        try:
            await bot.send_message(tg_id, f"Ваш заказ №{o.order_number} назначен курьеру 🚴")
        except Exception:
            pass

    await call.answer("✅ Курьер назначен")


# =========================
# COURIER ACTIONS
# =========================
async def is_registered_courier(session: AsyncSession, tg_id: int) -> bool:
    q = await session.execute(select(Courier).where(Courier.chat_id == tg_id, Courier.is_active == True))
    return q.scalar_one_or_none() is not None


@router_courier.callback_query(F.data.startswith("courier:accept:"))
async def courier_accept(call: CallbackQuery):
    order_id = int(call.data.split(":")[2])
    async with SessionLocal() as session:
        if not await is_registered_courier(session, call.from_user.id):
            await call.answer("Siz kuryer emassiz", show_alert=True)
            return

        o = await session.get(Order, order_id)
        if not o:
            await call.answer("Order not found", show_alert=True)
            return
        # only if assigned
        o.status = "OUT_FOR_DELIVERY"
        o.updated_at = utcnow()

        user = await session.get(User, o.user_id)
        items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == o.id))
        o.items = items_q.scalars().all()
        await send_or_edit_admin_post(session, o, user)
        await session.commit()

        tg_id = await get_user_tg_id(session, o.user_id)
        try:
            await bot.send_message(tg_id, f"Ваш заказ №{o.order_number} передан курьеру 🚴")
        except Exception:
            pass

    await call.answer("✅")


@router_courier.callback_query(F.data.startswith("courier:delivered:"))
async def courier_delivered(call: CallbackQuery):
    order_id = int(call.data.split(":")[2])
    async with SessionLocal() as session:
        if not await is_registered_courier(session, call.from_user.id):
            await call.answer("Siz kuryer emassiz", show_alert=True)
            return

        o = await session.get(Order, order_id)
        if not o:
            await call.answer("Order not found", show_alert=True)
            return

        o.status = "DELIVERED"
        o.updated_at = utcnow()
        o.delivered_at = utcnow()

        user = await session.get(User, o.user_id)
        items_q = await session.execute(select(OrderItem).where(OrderItem.order_id == o.id))
        o.items = items_q.scalars().all()

        # edit admin post: remove buttons
        await send_or_edit_admin_post(session, o, user)
        await session.commit()

        tg_id = await get_user_tg_id(session, o.user_id)
        try:
            await bot.send_message(tg_id, f"Ваш заказ №{o.order_number} успешно доставлен 🎉 Спасибо!")
        except Exception:
            pass

    # remove inline from courier message as well
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await call.answer("✅ Доставлен")


# =========================
# FASTAPI (API + WEBHOOK)
# =========================
app = FastAPI(title="FIESTA Backend")

# CORS (WebApp on Vercel + Telegram webview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[WEBAPP_URL, "https://web.telegram.org", "https://t.me", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"service": "FIESTA", "ok": True, "health": "/api/health"}


@app.get("/api/health")
async def health():
    return {"ok": True, "ts": utcnow().isoformat()}


def require_webapp_user(x_tg_init_data: str) -> Dict[str, Any]:
    try:
        return verify_telegram_init_data(x_tg_init_data, BOT_TOKEN)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"initData invalid: {e}")


@app.get("/api/categories")
async def api_categories(x_tg_init_data: str = Header(default="")):
    _user = require_webapp_user(x_tg_init_data)
    async with SessionLocal() as session:
        q = await session.execute(select(Category).where(Category.is_active == True).order_by(asc(Category.name)))
        cats = q.scalars().all()
    # add virtual "All"
    data = [{"id": 0, "name": "All"}] + [{"id": c.id, "name": c.name} for c in cats]
    return {"categories": data}


@app.get("/api/foods")
async def api_foods(
    x_tg_init_data: str = Header(default=""),
    q: str = "",
    category_id: int = 0,
    sort: str = "new"  # "rating" | "new" | "price_asc" | "price_desc"
):
    _user = require_webapp_user(x_tg_init_data)
    async with SessionLocal() as session:
        stmt = select(Food).where(Food.is_active == True)
        if category_id and category_id != 0:
            stmt = stmt.where(Food.category_id == category_id)
        if q:
            like = f"%{q.strip()}%"
            stmt = stmt.where(Food.name.ilike(like))

        if sort == "rating":
            stmt = stmt.order_by(desc(Food.rating))
        elif sort == "price_asc":
            stmt = stmt.order_by(asc(Food.price))
        elif sort == "price_desc":
            stmt = stmt.order_by(desc(Food.price))
        else:
            stmt = stmt.order_by(desc(Food.created_at))

        res = await session.execute(stmt.limit(200))
        foods = res.scalars().all()

    return {
        "foods": [
            {
                "id": f.id,
                "category_id": f.category_id,
                "name": f.name,
                "description": f.description,
                "price": f.price,
                "rating": f.rating,
                "is_new": f.is_new,
                "image_url": f.image_url,
                "created_at": f.created_at.isoformat(),
            }
            for f in foods
        ]
    }


@app.get("/api/promo/validate")
async def api_promo_validate(x_tg_init_data: str = Header(default=""), code: str = ""):
    _user = require_webapp_user(x_tg_init_data)
    code = (code or "").strip().upper()
    if not code:
        return {"ok": False, "reason": "empty"}

    async with SessionLocal() as session:
        q = await session.execute(select(Promo).where(Promo.code == code))
        p = q.scalar_one_or_none()
        if not p or not p.is_active:
            return {"ok": False, "reason": "not_found"}
        if p.expires_at and p.expires_at < utcnow():
            return {"ok": False, "reason": "expired"}
        if p.used_count >= p.usage_limit:
            return {"ok": False, "reason": "limit"}
        return {"ok": True, "discount_percent": p.discount_percent}


# Telegram webhook endpoint
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return JSONResponse({"ok": True})


# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def on_startup():
    # create tables (simple alternative to alembic inside 2-file constraint)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as session:
        await seed_demo_data(session)
        await session.commit()

    webhook_url = f"{API_PUBLIC_BASE}/telegram/webhook"
    try:
        await bot.set_webhook(webhook_url, drop_pending_updates=True)
        me = await bot.get_me()
        log.info(f"Webhook set: {webhook_url} | bot=@{me.username}")
    except Exception as e:
        log.error(f"set_webhook failed: {e}")


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await bot.session.close()
    except Exception:
        pass
    await engine.dispose()
