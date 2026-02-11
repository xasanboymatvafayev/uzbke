"""
FIESTA Food Delivery — single-file production-ready-ish demo (Bot + API) for Python 3.11+

Stack:
- aiogram 3.x (async)
- FastAPI (async)
- SQLAlchemy async + asyncpg
- Redis (FSM storage)
- Single file by user request (main.py) + single WebApp file (web_app/index.html)

⚠️ SECURITY NOTE:
- Do NOT hardcode secrets here. Use environment variables.
- If you pasted your real BOT_TOKEN / DB creds publicly, rotate them immediately.

ENV:
BOT_TOKEN=...
ADMIN_IDS=123,456
DB_URL=postgresql+asyncpg://...
REDIS_URL=redis://...
SHOP_CHANNEL_ID=-100...
COURIER_CHANNEL_ID=-100...
WEBAPP_URL=https://...
API_HOST=0.0.0.0
API_PORT=8000
BOT_USERNAME=auto (fetched via getMe on startup)

Run (local):
  python main.py

Docker/compose, alembic files, etc. are omitted ONLY because you required only 2 files.
You can still add them later without changing the architecture.
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import (
    CallbackQuery,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    WebAppInfo,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ---------------------------
# Logging
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("fiesta")

# ---------------------------
# Config
# ---------------------------
def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env: {name}")
    return v

BOT_TOKEN = os.getenv("BOT_TOKEN")
DB_URL = os.getenv("DB_URL")
REDIS_URL = os.getenv("REDIS_URL")
WEBAPP_URL = os.getenv("WEBAPP_URL")

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ADMIN IDS (comma orqali: 6365371142,123456789)
ADMIN_IDS = {
    int(x.strip())
    for x in os.getenv("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}

# Optional channel IDs (bo‘lmasa None bo‘ladi)
SHOP_CHANNEL_ID = (
    int(os.getenv("SHOP_CHANNEL_ID"))
    if os.getenv("SHOP_CHANNEL_ID")
    else None
)

COURIER_CHANNEL_ID = (
    int(os.getenv("COURIER_CHANNEL_ID"))
    if os.getenv("COURIER_CHANNEL_ID")
    else None
)


# ---------------------------
# Database
# ---------------------------
engine = create_async_engine(DB_URL, pool_pre_ping=True)
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class OrderStatus(StrEnum):
    NEW = "NEW"  # Принят
    CONFIRMED = "CONFIRMED"  # Подтвержден
    COOKING = "COOKING"  # Готовится
    COURIER_ASSIGNED = "COURIER_ASSIGNED"  # Курьер назначен
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"  # Передан курьеру
    DELIVERED = "DELIVERED"  # Доставлен
    CANCELED = "CANCELED"  # Отменен

STATUS_LABEL = {
    OrderStatus.NEW: "Принят",
    OrderStatus.CONFIRMED: "Подтвержден",
    OrderStatus.COOKING: "Готовится",
    OrderStatus.COURIER_ASSIGNED: "Курьер назначен",
    OrderStatus.OUT_FOR_DELIVERY: "Передан курьеру",
    OrderStatus.DELIVERED: "Доставлен",
    OrderStatus.CANCELED: "Отменен",
}

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tg_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    full_name: Mapped[str] = mapped_column(String(256))
    joined_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    ref_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    promo_given_15: Mapped[bool] = mapped_column(Boolean, default=False)

    orders: Mapped[List["Order"]] = relationship(back_populates="user", cascade="all,delete-orphan")

class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    foods: Mapped[List["Food"]] = relationship(back_populates="category", cascade="all,delete-orphan")

class Food(Base):
    __tablename__ = "foods"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(String(512), default="")
    price: Mapped[int] = mapped_column(Integer)  # sums
    rating: Mapped[float] = mapped_column(Numeric(3, 2), default=5.0)
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

    category: Mapped["Category"] = relationship(back_populates="foods")

class Courier(Base):
    __tablename__ = "couriers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())

class Promo(Base):
    __tablename__ = "promos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    discount_percent: Mapped[int] = mapped_column(Integer)  # 1-90
    expires_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    usage_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    used_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_by_user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_number: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    customer_name: Mapped[str] = mapped_column(String(128))
    phone: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(String(512), default="")
    total: Mapped[int] = mapped_column(Integer)
    status: Mapped[OrderStatus] = mapped_column(Enum(OrderStatus, name="order_status"), default=OrderStatus.NEW)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    delivered_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    location_lat: Mapped[float] = mapped_column(Numeric(10, 6))
    location_lng: Mapped[float] = mapped_column(Numeric(10, 6))

    courier_id: Mapped[Optional[int]] = mapped_column(ForeignKey("couriers.id"), nullable=True)
    admin_channel_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    courier_message_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # in courier channel (if used)

    user: Mapped["User"] = relationship(back_populates="orders")
    items: Mapped[List["OrderItem"]] = relationship(back_populates="order", cascade="all,delete-orphan")
    courier: Mapped[Optional["Courier"]] = relationship()

class OrderItem(Base):
    __tablename__ = "order_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    food_id: Mapped[int] = mapped_column(Integer)
    name_snapshot: Mapped[str] = mapped_column(String(128))
    price_snapshot: Mapped[int] = mapped_column(Integer)
    qty: Mapped[int] = mapped_column(Integer)
    line_total: Mapped[int] = mapped_column(Integer)

    order: Mapped["Order"] = relationship(back_populates="items")

Index("ix_orders_status_created", Order.status, Order.created_at)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # seed demo categories & foods if empty
    async with SessionLocal() as s:
        res = await s.execute(select(func.count(Category.id)))
        if res.scalar_one() == 0:
            cat_names = ["Lavash","Burger","Xaggi","Shaurma","Hotdog","Combo","Sneki","Sous","Napitki"]
            cats = [Category(name=n, is_active=True) for n in cat_names]
            s.add_all(cats)
            await s.flush()
            # 3 foods each
            demo = []
            for c in cats:
                for i in range(1, 4):
                    demo.append(Food(
                        category_id=c.id,
                        name=f"{c.name} #{i}",
                        description=f"Вкусный {c.name.lower()} — demo {i}",
                        price=25000 + i*5000,
                        rating=4.5 + (i*0.1),
                        is_new=(i == 3),
                        is_active=True,
                        image_url=None,
                    ))
            s.add_all(demo)
            await s.commit()
            log.info("Seeded demo data.")

# ---------------------------
# Helpers
# ---------------------------
def is_admin(tg_id: int) -> bool:
    return tg_id in ADMIN_IDS

def now_tz() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def short_order_number() -> str:
    # 9-10 chars
    return secrets.token_hex(5)[:10].upper()

def map_link(lat: float, lng: float) -> str:
    return f"https://maps.google.com/?q={lat},{lng}"

def phone_sanitize(p: str) -> str:
    p = re.sub(r"[^\d+]", "", p)
    return p[:32]

# ---------------------------
# Telegram initData verification (FastAPI)
# ---------------------------
def verify_telegram_init_data(init_data: str, bot_token: str) -> dict:
    """
    Verifies Telegram WebApp initData according to Telegram docs:
    - Parse key-value pairs from querystring
    - Exclude "hash", sort by key, join as "k=v\n..."
    - secret_key = HMAC_SHA256("WebAppData", bot_token)
    - expected_hash = HMAC_SHA256(secret_key, data_check_string) hex
    """
    if not init_data:
        raise ValueError("empty init_data")
    pairs = []
    for part in init_data.split("&"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        pairs.append((k, v))
    data = dict(pairs)
    recv_hash = data.pop("hash", None)
    if not recv_hash:
        raise ValueError("missing hash")
    data_check = "\n".join(f"{k}={data[k]}" for k in sorted(data.keys()))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    expected = hmac.new(secret_key, data_check.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, recv_hash):
        raise ValueError("bad hash")
    # user is JSON in data["user"]
    user_json = data.get("user")
    user = json.loads(_url_decode(user_json)) if user_json else None
    return {"raw": data, "user": user}

def _url_decode(s: str) -> str:
    from urllib.parse import unquote_plus
    return unquote_plus(s)

# ---------------------------
# Services
# ---------------------------
class UsersService:
    @staticmethod
    async def get_or_create_user(session: AsyncSession, tg_id: int, username: str | None, full_name: str, ref_by_tg_id: Optional[int]) -> User:
        q = await session.execute(select(User).where(User.tg_id == tg_id))
        u = q.scalar_one_or_none()
        if u:
            # update profile changes
            changed = False
            if u.username != username:
                u.username = username
                changed = True
            if u.full_name != full_name:
                u.full_name = full_name
                changed = True
            if changed:
                await session.commit()
            return u

        ref_by_user_id = None
        if ref_by_tg_id and ref_by_tg_id != tg_id:
            rq = await session.execute(select(User).where(User.tg_id == ref_by_tg_id))
            ru = rq.scalar_one_or_none()
            if ru:
                ref_by_user_id = ru.id

        u = User(tg_id=tg_id, username=username, full_name=full_name, joined_at=now_tz(), ref_by_user_id=ref_by_user_id)
        session.add(u)
        await session.commit()
        return u

class ReferralService:
    @staticmethod
    async def referral_stats(session: AsyncSession, user_id: int) -> dict:
        ref_count = (await session.execute(select(func.count(User.id)).where(User.ref_by_user_id == user_id))).scalar_one()
        orders_count = (await session.execute(select(func.count(Order.id)).where(Order.user_id == user_id))).scalar_one()
        delivered_count = (await session.execute(select(func.count(Order.id)).where(Order.user_id == user_id, Order.status == OrderStatus.DELIVERED))).scalar_one()
        return {
            "ref_count": int(ref_count),
            "orders_count": int(orders_count),
            "paid_or_delivered_count": int(delivered_count),
        }

    @staticmethod
    async def maybe_grant_ref_promo(session: AsyncSession, user: User) -> Optional[Promo]:
        stats = await ReferralService.referral_stats(session, user.id)
        if stats["ref_count"] >= 3 and not user.promo_given_15:
            code = f"REF{user.id}{secrets.token_hex(2).upper()}"
            promo = Promo(code=code, discount_percent=15, is_active=True, created_by_user_id=user.id)
            session.add(promo)
            user.promo_given_15 = True
            await session.commit()
            return promo
        return None

class FoodsService:
    @staticmethod
    async def list_categories(session: AsyncSession) -> List[dict]:
        rows = (await session.execute(select(Category).where(Category.is_active == True).order_by(Category.name))).scalars().all()
        return [{"id": c.id, "name": c.name} for c in rows]

    @staticmethod
    async def list_foods(session: AsyncSession) -> List[dict]:
        rows = (await session.execute(select(Food).where(Food.is_active == True))).scalars().all()
        return [{
            "id": f.id,
            "category_id": f.category_id,
            "name": f.name,
            "description": f.description,
            "price": int(f.price),
            "rating": float(f.rating),
            "is_new": bool(f.is_new),
            "image_url": f.image_url,
            "created_at": f.created_at.isoformat(),
        } for f in rows]

class PromoService:
    @staticmethod
    async def validate_code(session: AsyncSession, code: str) -> dict:
        code = (code or "").strip().upper()
        if not code:
            return {"ok": False, "reason": "empty"}
        promo = (await session.execute(select(Promo).where(Promo.code == code, Promo.is_active == True))).scalar_one_or_none()
        if not promo:
            return {"ok": False, "reason": "not_found"}
        if promo.expires_at and promo.expires_at < now_tz():
            return {"ok": False, "reason": "expired"}
        if promo.usage_limit is not None and promo.used_count >= promo.usage_limit:
            return {"ok": False, "reason": "limit"}
        return {"ok": True, "discount_percent": int(promo.discount_percent)}

    @staticmethod
    async def consume(session: AsyncSession, code: Optional[str]) -> Optional[Promo]:
        if not code:
            return None
        code = code.strip().upper()
        promo = (await session.execute(select(Promo).where(Promo.code == code, Promo.is_active == True))).scalar_one_or_none()
        if not promo:
            return None
        if promo.expires_at and promo.expires_at < now_tz():
            return None
        if promo.usage_limit is not None and promo.used_count >= promo.usage_limit:
            return None
        promo.used_count += 1
        await session.commit()
        return promo

class OrdersService:
    @staticmethod
    async def create_order(session: AsyncSession, user: User, payload: dict) -> Order:
        items = payload.get("items") or []
        total = int(payload.get("total") or 0)
        if total < 50_000:
            raise ValueError("min_total")
        customer_name = (payload.get("customer_name") or user.full_name)[:128]
        phone = phone_sanitize(payload.get("phone") or "")
        if len(phone) < 7:
            raise ValueError("phone")
        comment = (payload.get("comment") or "")[:512]
        loc = payload.get("location") or {}
        lat = float(loc.get("lat"))
        lng = float(loc.get("lng"))
        promo_code = (payload.get("promo_code") or "").strip().upper() or None

        # Optional promo validation & apply discount client side; server can re-check
        promo = None
        if promo_code:
            promo_ok = await PromoService.validate_code(session, promo_code)
            if promo_ok.get("ok"):
                promo = await PromoService.consume(session, promo_code)

        # Recalculate line totals from snapshot in payload (trusted only partially). In real prod: fetch prices from DB.
        order_number = short_order_number()
        o = Order(
            order_number=order_number,
            user_id=user.id,
            customer_name=customer_name,
            phone=phone,
            comment=comment,
            total=total,
            status=OrderStatus.NEW,
            created_at=now_tz(),
            location_lat=lat,
            location_lng=lng,
        )
        session.add(o)
        await session.flush()

        oi_list: List[OrderItem] = []
        for it in items:
            fid = int(it.get("food_id"))
            name = str(it.get("name") or "Item")[:128]
            qty = max(1, int(it.get("qty") or 1))
            price = int(it.get("price") or 0)
            line_total = qty * price
            oi_list.append(OrderItem(
                order_id=o.id, food_id=fid, name_snapshot=name,
                price_snapshot=price, qty=qty, line_total=line_total
            ))
        session.add_all(oi_list)
        await session.commit()
        await session.refresh(o)
        return o

    @staticmethod
    async def list_last_orders(session: AsyncSession, user: User, limit: int = 10) -> List[Order]:
        rows = (await session.execute(
            select(Order).where(Order.user_id == user.id).order_by(Order.created_at.desc()).limit(limit)
        )).scalars().all()
        # eager load items
        for o in rows:
            await session.refresh(o, attribute_names=["items"])
        return rows

    @staticmethod
    async def active_orders(session: AsyncSession) -> List[Order]:
        active = {
            OrderStatus.NEW, OrderStatus.CONFIRMED, OrderStatus.COOKING,
            OrderStatus.COURIER_ASSIGNED, OrderStatus.OUT_FOR_DELIVERY
        }
        rows = (await session.execute(select(Order).where(Order.status.in_(active)).order_by(Order.created_at.desc()))).scalars().all()
        for o in rows:
            await session.refresh(o, attribute_names=["items", "user", "courier"])
        return rows

    @staticmethod
    async def set_status(session: AsyncSession, order_id: int, status: OrderStatus, courier_id: Optional[int] = None) -> Order:
        o = (await session.execute(select(Order).where(Order.id == order_id))).scalar_one()
        o.status = status
        if courier_id is not None:
            o.courier_id = courier_id
        if status == OrderStatus.DELIVERED:
            o.delivered_at = now_tz()
        await session.commit()
        await session.refresh(o)
        await session.refresh(o, attribute_names=["items", "user", "courier"])
        return o

class StatsService:
    @staticmethod
    async def summary(session: AsyncSession, since: dt.datetime) -> dict:
        orders_count = (await session.execute(select(func.count(Order.id)).where(Order.created_at >= since))).scalar_one()
        delivered_count = (await session.execute(select(func.count(Order.id)).where(Order.created_at >= since, Order.status == OrderStatus.DELIVERED))).scalar_one()
        revenue_sum = (await session.execute(select(func.coalesce(func.sum(Order.total), 0)).where(Order.created_at >= since, Order.status == OrderStatus.DELIVERED))).scalar_one()
        active_count = (await session.execute(select(func.count(Order.id)).where(Order.status.in_([
            OrderStatus.NEW, OrderStatus.CONFIRMED, OrderStatus.COOKING, OrderStatus.COURIER_ASSIGNED, OrderStatus.OUT_FOR_DELIVERY
        ])))).scalar_one()
        return {
            "orders_count": int(orders_count),
            "delivered_count": int(delivered_count),
            "revenue_sum": int(revenue_sum),
            "active_orders": int(active_count),
        }

# ---------------------------
# Telegram notify / rendering
# ---------------------------
def build_client_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))],
            [KeyboardButton(text="📦 Мои заказы"), KeyboardButton(text="ℹ️ Информация о нас")],
            [KeyboardButton(text="👥 Пригласить друга")],
        ],
        resize_keyboard=True
    )

def shop_inline() -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="🛍 Заказать", web_app=WebAppInfo(url=WEBAPP_URL))
    return kb

def order_admin_kb(order_id: int) -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Подтвержден", callback_data=f"adm:status:{order_id}:{OrderStatus.CONFIRMED}")
    kb.button(text="🍳 Готовится", callback_data=f"adm:status:{order_id}:{OrderStatus.COOKING}")
    kb.button(text="🚴 Курьер", callback_data=f"adm:courier_pick:{order_id}")
    kb.button(text="❌ Отменен", callback_data=f"adm:status:{order_id}:{OrderStatus.CANCELED}")
    kb.adjust(2, 2)
    return kb

def courier_pick_kb(order_id: int, couriers: List[Courier]) -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    for c in couriers:
        kb.button(text=f"🚴 {c.name}", callback_data=f"adm:assign:{order_id}:{c.id}")
    kb.button(text="⬅️ Назад", callback_data=f"adm:back_to_order:{order_id}")
    kb.adjust(1)
    return kb

def courier_msg_kb(order_id: int) -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Qabul qildim", callback_data=f"cour:accept:{order_id}")
    kb.button(text="📦 Yetkazildi", callback_data=f"cour:delivered:{order_id}")
    kb.adjust(2)
    return kb

def fmt_order_items(o: Order) -> str:
    lines = []
    for it in o.items:
        lines.append(f"• {it.name_snapshot} x{it.qty} = {it.line_total} сум")
    return "\n".join(lines) if lines else "—"

def fmt_order_client(o: Order) -> str:
    created = o.created_at.astimezone(dt.timezone(dt.timedelta(hours=5))).strftime("%Y-%m-%d %H:%M")
    return (
        f"🆔 Заказ №{o.order_number} | {created}\n"
        f"💰 {o.total} сум | 📦 {STATUS_LABEL[o.status]}\n\n"
        f"{fmt_order_items(o)}"
    )

def fmt_order_admin(o: Order) -> str:
    created = o.created_at.astimezone(dt.timezone(dt.timedelta(hours=5))).strftime("%Y-%m-%d %H:%M")
    u = o.user
    uname = f"@{u.username}" if u.username else "—"
    lat = float(o.location_lat); lng = float(o.location_lng)
    courier = o.courier.name if o.courier else "—"
    return (
        f"🆕 Заказ №{o.order_number}\n"
        f"Статус: **{STATUS_LABEL[o.status]}**\n\n"
        f"👤 Пользователь: {u.full_name} ({uname})\n"
        f"📞 Телефон: {o.phone}\n"
        f"💰 Сумма: {o.total} сум\n"
        f"🕒 Время: {created}\n"
        f"🚴 Курьер: {courier}\n"
        f"📍 Локация: {lat},{lng}\n"
        f"🗺️ {map_link(lat,lng)}\n\n"
        f"🍽️ Заказ:\n{fmt_order_items(o)}\n\n"
        f"📍 Локация: {o.order_number} | Локация"
    )

def fmt_order_courier(o: Order) -> str:
    lat = float(o.location_lat); lng = float(o.location_lng)
    return (
        f"🚴 Новый заказ №{o.order_number}\n"
        f"👤 Клиент: {o.customer_name}\n"
        f"📞 Телефон: {o.phone}\n"
        f"💰 Сумма: {o.total} сум\n"
        f"📍 Локация: {map_link(lat,lng)}\n\n"
        f"🍽️ Список:\n{fmt_order_items(o)}"
    )

async def notify_user(bot: Bot, tg_id: int, text: str) -> None:
    try:
        await bot.send_message(tg_id, text)
    except Exception as e:
        log.warning("notify_user failed: %s", e)

async def post_or_edit_admin_order(bot: Bot, session: AsyncSession, o: Order) -> None:
    global SHOP_CHANNEL_ID
    if not SHOP_CHANNEL_ID:
        return
    text = fmt_order_admin(o)
    kb = None
    if o.status not in {OrderStatus.DELIVERED, OrderStatus.CANCELED}:
        kb = order_admin_kb(o.id).as_markup()
    try:
        if o.admin_channel_message_id:
            await bot.edit_message_text(
                chat_id=SHOP_CHANNEL_ID,
                message_id=o.admin_channel_message_id,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
        else:
            msg = await bot.send_message(
                chat_id=SHOP_CHANNEL_ID,
                text=text,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
            o.admin_channel_message_id = msg.message_id
            await session.commit()
    except Exception as e:
        log.warning("post_or_edit_admin_order failed: %s", e)

async def post_or_edit_courier_order(bot: Bot, session: AsyncSession, o: Order) -> None:
    # send to courier channel if exists; else to courier private chat_id
    text = fmt_order_courier(o)
    kb = None
    if o.status in {OrderStatus.COURIER_ASSIGNED, OrderStatus.OUT_FOR_DELIVERY}:
        kb = courier_msg_kb(o.id).as_markup()
    target_chat = COURIER_CHANNEL_ID or (o.courier.chat_id if o.courier else None)
    if not target_chat:
        return
    try:
        if o.courier_message_id and COURIER_CHANNEL_ID:
            await bot.edit_message_text(
                chat_id=target_chat,
                message_id=o.courier_message_id,
                text=text,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
        else:
            msg = await bot.send_message(
                chat_id=target_chat,
                text=text,
                reply_markup=kb,
                disable_web_page_preview=True,
            )
            if COURIER_CHANNEL_ID:
                o.courier_message_id = msg.message_id
                await session.commit()
    except Exception as e:
        log.warning("post_or_edit_courier_order failed: %s", e)

# ---------------------------
# FSM (Admin)
# ---------------------------
class AdminFoodFSM(StatesGroup):
    name = State()
    category_id = State()
    price = State()
    rating = State()
    is_new = State()
    image_url = State()
    description = State()

class AdminCourierFSM(StatesGroup):
    chat_id = State()
    name = State()

class AdminPromoFSM(StatesGroup):
    code = State()
    discount = State()
    expires = State()
    usage_limit = State()

class AdminSettingsFSM(StatesGroup):
    shop_channel_id = State()
    courier_channel_id = State()

# ---------------------------
# Routers
# ---------------------------
client_router = Router()
admin_router = Router()
courier_router = Router()

# ---------------------------
# Dependencies
# ---------------------------
async def get_session() -> AsyncSession:
    async with SessionLocal() as s:
        yield s

# ---------------------------
# Client handlers
# ---------------------------
@client_router.message(CommandStart())
async def start_cmd(message: Message, session: AsyncSession, bot: Bot):
    ref_by = None
    args = message.text.split(maxsplit=1)
    if len(args) == 2 and args[1].isdigit():
        ref_by = int(args[1])

    u = await UsersService.get_or_create_user(
        session=session,
        tg_id=message.from_user.id,
        username=message.from_user.username,
        full_name=message.from_user.full_name,
        ref_by_tg_id=ref_by
    )

    text = f"Добро пожаловать в FIESTA! {u.full_name}\nДля заказа перейдите по кнопке ➡️ 🛍 Заказать"
    await message.answer(text, reply_markup=build_client_menu())

@client_router.message(Command("shop"))
async def shop_cmd(message: Message):
    await message.answer("Чтобы открыть наш магазин, нажмите кнопку ниже", reply_markup=shop_inline().as_markup())

@client_router.message(F.text == "📦 Мои заказы")
async def my_orders(message: Message, session: AsyncSession):
    u = (await session.execute(select(User).where(User.tg_id == message.from_user.id))).scalar_one_or_none()
    if not u:
        await message.answer("Сначала нажмите /start")
        return
    orders = await OrdersService.list_last_orders(session, u, limit=10)
    if not orders:
        await message.answer("В данный момент у вас нет активных заказов в нашем магазине. Чтобы открыть магазин, введите команду — /shop")
        return
    for o in orders:
        await message.answer(fmt_order_client(o))

@client_router.message(F.text == "ℹ️ Информация о нас")
async def info_about(message: Message):
    await message.answer(
        "🌟 Добро Пожаловать в FIESTA !\n"
        "📍 Наш адрес:Хорезмская область, г.Хива, махаллинский сход граждан Гиламчи\n"
        "🏢﻿ Ориентир: Школа №12 Оруджева\n"
        "📞 Контактный номер: +998 91 420 15 15\n"
        "🕙﻿ Рабочие часы: 24/7\n"
        "📷 Мы в Instagram: fiesta.khiva (https://www.instagram.com/fiesta.khiva?igsh=Z3VoMzE0eGx0ZTVo)\n"
        "🔗 Найти нас на карте: Место расположение (https://maps.app.goo.gl/dpBVHBWX1K7NTYVR7)"
    )

@client_router.message(F.text == "👥 Пригласить друга")
async def invite_friend(message: Message, session: AsyncSession, bot: Bot):
    u = (await session.execute(select(User).where(User.tg_id == message.from_user.id))).scalar_one_or_none()
    if not u:
        await message.answer("Сначала нажмите /start")
        return

    # BOT_USERNAME is set on startup
    bot_me = await bot.get_me()
    bot_username = bot_me.username

    stats = await ReferralService.referral_stats(session, u.id)
    promo = await ReferralService.maybe_grant_ref_promo(session, u)

    text = (
        "За приглашение друга, вы можете получить промо-код от нас 👥\n"
        f"Вы пригласили {stats['ref_count']} человек\n"
        f"🛒 Оформили заказов: {stats['orders_count']}\n"
        f"💰 Оплатили заказов: {stats['paid_or_delivered_count']}\n"
        f"👤 Ваша реферальная ссылка: https://t.me/{bot_username}?start={u.tg_id}\n"
        "Пригласите трех человек и вы получите от нас промо-код со скидкой 15%"
    )
    await message.answer(text)
    if promo:
        await message.answer(f"🎁 Ваш промокод на 15%: **{promo.code}**", parse_mode=ParseMode.MARKDOWN)

# ---------------------------
# WebApp -> Bot order_create
# ---------------------------
@client_router.message(F.web_app_data)
async def webapp_data(message: Message, session: AsyncSession, bot: Bot):
    u = (await session.execute(select(User).where(User.tg_id == message.from_user.id))).scalar_one_or_none()
    if not u:
        await message.answer("Сначала нажмите /start")
        return

    try:
        payload = json.loads(message.web_app_data.data)
    except Exception:
        await message.answer("Ошибка данных WebApp.")
        return

    if payload.get("type") != "order_create":
        await message.answer("Неизвестный тип данных.")
        return

    try:
        order = await OrdersService.create_order(session, u, payload)
    except ValueError as e:
        if str(e) == "min_total":
            await message.answer("Минимальная сумма заказа — 50 000 сум.")
        elif str(e) == "phone":
            await message.answer("Укажите корректный номер телефона.")
        else:
            await message.answer("Ошибка создания заказа.")
        return
    except Exception as e:
        log.exception("create_order error: %s", e)
        await message.answer("Ошибка сервера. Попробуйте позже.")
        return

    await message.answer(
        f"Ваш заказ принят ✅\n🆔 Заказ №{order.order_number}\n💰 Сумма: {order.total} сум\n📦 Статус: Принят"
    )

    # Post to admin channel
    await post_or_edit_admin_order(bot, session, order)

# ---------------------------
# Admin panel
# ---------------------------
def admin_menu_kb() -> InlineKeyboardBuilder:
    kb = InlineKeyboardBuilder()
    kb.button(text="🍔 Taomlar", callback_data="adm:foods")
    kb.button(text="🚴 Kuryerlar", callback_data="adm:couriers")
    kb.button(text="🎁 Promokodlar", callback_data="adm:promos")
    kb.button(text="📊 Statistika", callback_data="adm:stats")
    kb.button(text="📦 Aktiv buyurtmalar", callback_data="adm:active_orders")
    kb.button(text="⚙️ Sozlamalar", callback_data="adm:settings")
    kb.adjust(2, 2, 2)
    return kb

@admin_router.message(Command("admin"))
async def admin_cmd(message: Message):
    if not is_admin(message.from_user.id):
        return
    await message.answer("Admin panel:", reply_markup=admin_menu_kb().as_markup())

@admin_router.callback_query(F.data == "adm:foods")
async def adm_foods(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Add food", callback_data="adm:food_add")
    kb.button(text="📃 List foods", callback_data="adm:food_list")
    kb.button(text="⬅️ Back", callback_data="adm:back")
    kb.adjust(1)
    await call.message.edit_text("🍔 Foods:", reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:food_list")
async def adm_food_list(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    rows = (await session.execute(select(Food).order_by(Food.created_at.desc()).limit(15))).scalars().all()
    if not rows:
        await call.message.edit_text("Foods empty.", reply_markup=admin_menu_kb().as_markup())
        return
    text = "Последние 15 таомов:\n\n" + "\n".join([f"#{f.id} {f.name} | {int(f.price)} | {'ON' if f.is_active else 'OFF'}" for f in rows])
    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Back", callback_data="adm:foods")
    await call.message.edit_text(text, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:food_add")
async def adm_food_add(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    await state.set_state(AdminFoodFSM.name)
    await call.message.answer("Food name?")
    await call.answer()

@admin_router.message(AdminFoodFSM.name)
async def adm_food_name(message: Message, state: FSMContext, session: AsyncSession):
    if not is_admin(message.from_user.id):
        return
    await state.update_data(name=message.text.strip()[:128])

    cats = (await session.execute(select(Category).where(Category.is_active == True))).scalars().all()
    if not cats:
        await message.answer("No categories. Create in DB.")
        await state.clear()
        return
    kb = InlineKeyboardBuilder()
    for c in cats:
        kb.button(text=c.name, callback_data=f"adm:food_cat:{c.id}")
    kb.adjust(2)
    await message.answer("Select category:", reply_markup=kb.as_markup())

@admin_router.callback_query(F.data.startswith("adm:food_cat:"))
async def adm_food_cat(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    cid = int(call.data.split(":")[-1])
    await state.update_data(category_id=cid)
    await state.set_state(AdminFoodFSM.price)
    await call.message.answer("Price (sum)?")
    await call.answer()

@admin_router.message(AdminFoodFSM.price)
async def adm_food_price(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    try:
        price = int(re.sub(r"\D", "", message.text))
    except Exception:
        return await message.answer("Enter integer price.")
    await state.update_data(price=price)
    await state.set_state(AdminFoodFSM.rating)
    await message.answer("Rating (e.g. 4.7)?")

@admin_router.message(AdminFoodFSM.rating)
async def adm_food_rating(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    try:
        rating = float(message.text.strip().replace(",", "."))
    except Exception:
        return await message.answer("Enter number rating.")
    await state.update_data(rating=rating)
    await state.set_state(AdminFoodFSM.is_new)
    await message.answer("Is new? (yes/no)")

@admin_router.message(AdminFoodFSM.is_new)
async def adm_food_is_new(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    t = (message.text or "").lower()
    await state.update_data(is_new=t in {"y","yes","ha","true","1"})
    await state.set_state(AdminFoodFSM.image_url)
    await message.answer("Image URL (or '-' to skip)?")

@admin_router.message(AdminFoodFSM.image_url)
async def adm_food_image(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    url = message.text.strip()
    if url == "-":
        url = None
    await state.update_data(image_url=url)
    await state.set_state(AdminFoodFSM.description)
    await message.answer("Short description?")

@admin_router.message(AdminFoodFSM.description)
async def adm_food_desc(message: Message, state: FSMContext, session: AsyncSession):
    if not is_admin(message.from_user.id):
        return
    data = await state.get_data()
    desc = (message.text or "")[:512]
    f = Food(
        category_id=data["category_id"],
        name=data["name"],
        description=desc,
        price=int(data["price"]),
        rating=float(data["rating"]),
        is_new=bool(data["is_new"]),
        is_active=True,
        image_url=data.get("image_url"),
        created_at=now_tz()
    )
    session.add(f)
    await session.commit()
    await state.clear()
    await message.answer("✅ Food created.", reply_markup=admin_menu_kb().as_markup())

@admin_router.callback_query(F.data == "adm:couriers")
async def adm_couriers(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Add courier", callback_data="adm:courier_add")
    kb.button(text="📃 List couriers", callback_data="adm:courier_list")
    kb.button(text="⬅️ Back", callback_data="adm:back")
    kb.adjust(1)
    await call.message.edit_text("🚴 Couriers:", reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:courier_list")
async def adm_courier_list(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    rows = (await session.execute(select(Courier).order_by(Courier.created_at.desc()))).scalars().all()
    if not rows:
        await call.message.edit_text("Couriers empty.", reply_markup=admin_menu_kb().as_markup())
        return
    text = "Couriers:\n\n" + "\n".join([f"#{c.id} {c.name} | chat_id={c.chat_id} | {'ON' if c.is_active else 'OFF'}" for c in rows])
    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Back", callback_data="adm:couriers")
    await call.message.edit_text(text, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:courier_add")
async def adm_courier_add(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    await state.set_state(AdminCourierFSM.chat_id)
    await call.message.answer("Courier chat_id (numeric)?")
    await call.answer()

@admin_router.message(AdminCourierFSM.chat_id)
async def adm_courier_chatid(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    try:
        chat_id = int(message.text.strip())
    except Exception:
        return await message.answer("Enter numeric chat_id.")
    await state.update_data(chat_id=chat_id)
    await state.set_state(AdminCourierFSM.name)
    await message.answer("Courier name?")

@admin_router.message(AdminCourierFSM.name)
async def adm_courier_name(message: Message, state: FSMContext, session: AsyncSession):
    if not is_admin(message.from_user.id):
        return
    data = await state.get_data()
    c = Courier(chat_id=int(data["chat_id"]), name=message.text.strip()[:128], is_active=True, created_at=now_tz())
    session.add(c)
    await session.commit()
    await state.clear()
    await message.answer("✅ Courier added.", reply_markup=admin_menu_kb().as_markup())

@admin_router.callback_query(F.data == "adm:promos")
async def adm_promos(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Create promo", callback_data="adm:promo_add")
    kb.button(text="📃 List promos", callback_data="adm:promo_list")
    kb.button(text="⬅️ Back", callback_data="adm:back")
    kb.adjust(1)
    await call.message.edit_text("🎁 Promos:", reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:promo_list")
async def adm_promo_list(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    rows = (await session.execute(select(Promo).order_by(Promo.id.desc()).limit(20))).scalars().all()
    if not rows:
        await call.message.edit_text("Promos empty.", reply_markup=admin_menu_kb().as_markup())
        return
    def fmt(p: Promo) -> str:
        exp = p.expires_at.astimezone(dt.timezone(dt.timedelta(hours=5))).strftime("%Y-%m-%d") if p.expires_at else "—"
        lim = p.usage_limit if p.usage_limit is not None else "∞"
        return f"{p.code} | -{p.discount_percent}% | exp:{exp} | used:{p.used_count}/{lim} | {'ON' if p.is_active else 'OFF'}"
    text = "Promos:\n\n" + "\n".join(fmt(p) for p in rows)
    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Back", callback_data="adm:promos")
    await call.message.edit_text(text, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:promo_add")
async def adm_promo_add(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    await state.set_state(AdminPromoFSM.code)
    await call.message.answer("Promo code (leave '-' for random)?")
    await call.answer()

@admin_router.message(AdminPromoFSM.code)
async def adm_promo_code(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    code = message.text.strip().upper()
    if code == "-":
        code = f"SALE{secrets.token_hex(3).upper()}"
    await state.update_data(code=code)
    await state.set_state(AdminPromoFSM.discount)
    await message.answer("Discount percent (1-90)?")

@admin_router.message(AdminPromoFSM.discount)
async def adm_promo_discount(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    try:
        disc = int(re.sub(r"\D", "", message.text))
    except Exception:
        return await message.answer("Enter integer.")
    if disc < 1 or disc > 90:
        return await message.answer("1..90")
    await state.update_data(discount=disc)
    await state.set_state(AdminPromoFSM.expires)
    await message.answer("Expires at (YYYY-MM-DD) or '-'?")

@admin_router.message(AdminPromoFSM.expires)
async def adm_promo_expires(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    t = message.text.strip()
    exp = None
    if t != "-":
        try:
            y,m,d = map(int, t.split("-"))
            exp = dt.datetime(y,m,d,23,59,59,tzinfo=dt.timezone.utc)
        except Exception:
            return await message.answer("Format YYYY-MM-DD or '-'")
    await state.update_data(expires=exp)
    await state.set_state(AdminPromoFSM.usage_limit)
    await message.answer("Usage limit (int) or '-' for unlimited?")

@admin_router.message(AdminPromoFSM.usage_limit)
async def adm_promo_limit(message: Message, state: FSMContext, session: AsyncSession):
    if not is_admin(message.from_user.id):
        return
    t = message.text.strip()
    lim = None
    if t != "-":
        try:
            lim = int(re.sub(r"\D", "", t))
        except Exception:
            return await message.answer("Enter int or '-'")
    data = await state.get_data()
    promo = Promo(
        code=data["code"],
        discount_percent=int(data["discount"]),
        expires_at=data["expires"],
        usage_limit=lim,
        is_active=True,
        created_by_user_id=None
    )
    session.add(promo)
    await session.commit()
    await state.clear()
    await message.answer(f"✅ Promo created: {promo.code}", reply_markup=admin_menu_kb().as_markup())

@admin_router.callback_query(F.data == "adm:stats")
async def adm_stats(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    now = now_tz()
    day = now - dt.timedelta(days=1)
    week = now - dt.timedelta(days=7)
    month = now - dt.timedelta(days=30)
    s_day = await StatsService.summary(session, day)
    s_week = await StatsService.summary(session, week)
    s_month = await StatsService.summary(session, month)
    text = (
        f"📊 Statistika\n\n"
        f"Bugun (24h): заказов={s_day['orders_count']}, доставлено={s_day['delivered_count']}, выручка={s_day['revenue_sum']} сум\n"
        f"Hafta: заказов={s_week['orders_count']}, доставлено={s_week['delivered_count']}, выручка={s_week['revenue_sum']} сум\n"
        f"Oy: заказов={s_month['orders_count']}, доставлено={s_month['delivered_count']}, выручка={s_month['revenue_sum']} сум\n\n"
        f"Активные заказы сейчас: {s_month['active_orders']}"
    )
    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Back", callback_data="adm:back")
    await call.message.edit_text(text, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:active_orders")
async def adm_active_orders(call: CallbackQuery, session: AsyncSession):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    orders = await OrdersService.active_orders(session)
    if not orders:
        kb = InlineKeyboardBuilder()
        kb.button(text="⬅️ Back", callback_data="adm:back")
        await call.message.edit_text("Активных заказов нет.", reply_markup=kb.as_markup())
        return
    text = "📦 Активные заказы:\n\n" + "\n".join(
        [f"#{o.id} №{o.order_number} | {STATUS_LABEL[o.status]} | {o.total} сум" for o in orders[:20]]
    )
    kb = InlineKeyboardBuilder()
    for o in orders[:10]:
        kb.button(text=f"Open №{o.order_number}", callback_data=f"adm:open_order:{o.id}")
    kb.button(text="⬅️ Back", callback_data="adm:back")
    kb.adjust(1)
    await call.message.edit_text(text, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data.startswith("adm:open_order:"))
async def adm_open_order(call: CallbackQuery, session: AsyncSession, bot: Bot):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    oid = int(call.data.split(":")[-1])
    o = (await session.execute(select(Order).where(Order.id == oid))).scalar_one()
    await session.refresh(o, attribute_names=["items", "user", "courier"])
    text = fmt_order_admin(o)
    kb = InlineKeyboardBuilder()
    if o.admin_channel_message_id and SHOP_CHANNEL_ID:
        kb.button(text="🔗 Post", url=f"https://t.me/c/{str(abs(SHOP_CHANNEL_ID))[3:]}/{o.admin_channel_message_id}")
    kb.button(text="⬅️ Back", callback_data="adm:active_orders")
    await call.message.edit_text(text, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True, reply_markup=kb.as_markup())
    await call.answer()

@admin_router.callback_query(F.data == "adm:settings")
async def adm_settings(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    global SHOP_CHANNEL_ID, COURIER_CHANNEL_ID
    text = (
        "⚙️ Settings\n\n"
        f"SHOP_CHANNEL_ID: {SHOP_CHANNEL_ID}\n"
        f"COURIER_CHANNEL_ID: {COURIER_CHANNEL_ID}\n\n"
        "Send new SHOP_CHANNEL_ID (or '-'):"
    )
    await state.set_state(AdminSettingsFSM.shop_channel_id)
    await call.message.answer(text)
    await call.answer()

@admin_router.message(AdminSettingsFSM.shop_channel_id)
async def adm_set_shop_channel(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    global SHOP_CHANNEL_ID
    t = message.text.strip()
    if t != "-":
        try:
            SHOP_CHANNEL_ID = int(t)
        except Exception:
            return await message.answer("Enter numeric or '-'")
    await state.set_state(AdminSettingsFSM.courier_channel_id)
    await message.answer("Send new COURIER_CHANNEL_ID (or '-'):")

@admin_router.message(AdminSettingsFSM.courier_channel_id)
async def adm_set_courier_channel(message: Message, state: FSMContext):
    if not is_admin(message.from_user.id):
        return
    global COURIER_CHANNEL_ID
    t = message.text.strip()
    if t != "-":
        try:
            COURIER_CHANNEL_ID = int(t)
        except Exception:
            return await message.answer("Enter numeric or '-'")
    await state.clear()
    await message.answer("✅ Updated settings (runtime only).", reply_markup=admin_menu_kb().as_markup())

@admin_router.callback_query(F.data == "adm:back")
async def adm_back(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    await call.message.edit_text("Admin panel:", reply_markup=admin_menu_kb().as_markup())
    await call.answer()

# ---------------------------
# Admin channel callbacks: status + courier
# ---------------------------
@admin_router.callback_query(F.data.startswith("adm:status:"))
async def adm_set_status(call: CallbackQuery, session: AsyncSession, bot: Bot):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    _, _, oid_s, st_s = call.data.split(":", 3)
    oid = int(oid_s)
    status = OrderStatus(st_s)

    o = await OrdersService.set_status(session, oid, status=status)
    await post_or_edit_admin_order(bot, session, o)

    # notify user
    if status == OrderStatus.CONFIRMED:
        await notify_user(bot, o.user.tg_id, f"Ваш заказ №{o.order_number} подтвержден ✅")
    elif status == OrderStatus.COOKING:
        await notify_user(bot, o.user.tg_id, f"Ваш заказ №{o.order_number} готовится 🍳")
    elif status == OrderStatus.CANCELED:
        await notify_user(bot, o.user.tg_id, f"Ваш заказ №{o.order_number} отменен ❌")

    await call.answer("OK")

@admin_router.callback_query(F.data.startswith("adm:courier_pick:"))
async def adm_courier_pick(call: CallbackQuery, session: AsyncSession, bot: Bot):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    oid = int(call.data.split(":")[-1])
    couriers = (await session.execute(select(Courier).where(Courier.is_active == True).order_by(Courier.name))).scalars().all()
    if not couriers:
        return await call.answer("No couriers in DB", show_alert=True)

    kb = courier_pick_kb(oid, couriers).as_markup()
    await call.message.edit_text(f"Выберите курьера для заказа #{oid}", reply_markup=kb)
    await call.answer()

@admin_router.callback_query(F.data.startswith("adm:assign:"))
async def adm_assign_courier(call: CallbackQuery, session: AsyncSession, bot: Bot):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    _, _, oid_s, cid_s = call.data.split(":")
    oid = int(oid_s); cid = int(cid_s)

    o = await OrdersService.set_status(session, oid, status=OrderStatus.COURIER_ASSIGNED, courier_id=cid)
    # load courier relation
    await session.refresh(o, attribute_names=["items", "user", "courier"])

    await post_or_edit_admin_order(bot, session, o)
    await post_or_edit_courier_order(bot, session, o)
    await notify_user(bot, o.user.tg_id, f"К заказу №{o.order_number} назначен курьер 🚴")

    await call.answer("Courier assigned")

@admin_router.callback_query(F.data.startswith("adm:back_to_order:"))
async def adm_back_to_order(call: CallbackQuery, session: AsyncSession, bot: Bot):
    if not is_admin(call.from_user.id):
        return await call.answer("No access", show_alert=True)
    oid = int(call.data.split(":")[-1])
    o = (await session.execute(select(Order).where(Order.id == oid))).scalar_one()
    await session.refresh(o, attribute_names=["items", "user", "courier"])
    await call.message.edit_text(fmt_order_admin(o), parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True, reply_markup=order_admin_kb(oid).as_markup())
    await call.answer()

# ---------------------------
# Courier callbacks (works in courier channel OR private courier chat)
# ---------------------------
async def ensure_courier(session: AsyncSession, tg_id: int) -> Optional[Courier]:
    return (await session.execute(select(Courier).where(Courier.chat_id == tg_id, Courier.is_active == True))).scalar_one_or_none()

@courier_router.callback_query(F.data.startswith("cour:accept:"))
async def courier_accept(call: CallbackQuery, session: AsyncSession, bot: Bot):
    courier = await ensure_courier(session, call.from_user.id)
    if not courier:
        return await call.answer("Not courier", show_alert=True)
    oid = int(call.data.split(":")[-1])

    o = (await session.execute(select(Order).where(Order.id == oid))).scalar_one()
    if o.courier_id and o.courier_id != courier.id:
        return await call.answer("Assigned to another courier", show_alert=True)

    o = await OrdersService.set_status(session, oid, status=OrderStatus.OUT_FOR_DELIVERY, courier_id=courier.id)
    await post_or_edit_admin_order(bot, session, o)
    await post_or_edit_courier_order(bot, session, o)
    await notify_user(bot, o.user.tg_id, f"Ваш заказ №{o.order_number} передан курьеру 🚴")
    await call.answer("Accepted")

@courier_router.callback_query(F.data.startswith("cour:delivered:"))
async def courier_delivered(call: CallbackQuery, session: AsyncSession, bot: Bot):
    courier = await ensure_courier(session, call.from_user.id)
    if not courier:
        return await call.answer("Not courier", show_alert=True)
    oid = int(call.data.split(":")[-1])

    o = (await session.execute(select(Order).where(Order.id == oid))).scalar_one()
    if o.courier_id != courier.id:
        return await call.answer("Not your order", show_alert=True)

    o = await OrdersService.set_status(session, oid, status=OrderStatus.DELIVERED, courier_id=courier.id)
    # remove inline buttons on admin post by editing
    await post_or_edit_admin_order(bot, session, o)
    # courier side: update to delivered (remove buttons)
    try:
        target = COURIER_CHANNEL_ID or courier.chat_id
        if target and o.courier_message_id and COURIER_CHANNEL_ID:
            await bot.edit_message_reply_markup(chat_id=target, message_id=o.courier_message_id, reply_markup=None)
        else:
            # if private chat, just send final message
            await bot.send_message(courier.chat_id, f"✅ Заказ №{o.order_number} отмечен как доставлен.")
    except Exception:
        pass

    await notify_user(bot, o.user.tg_id, f"Ваш заказ №{o.order_number} успешно доставлен 🎉 Спасибо!")
    await call.answer("Delivered")

# ---------------------------
# FastAPI mini backend
# ---------------------------
api = FastAPI(title="FIESTA API")

async def api_session() -> AsyncSession:
    async with SessionLocal() as s:
        yield s

def get_init_data(request: Request) -> str:
    # Telegram passes initData in window.Telegram.WebApp.initData, we send it as header X-TG-INITDATA
    init_data = request.headers.get("X-TG-INITDATA") or ""
    return init_data

@api.middleware("http")
async def add_cors_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp

@api.options("/{path:path}")
async def options_any(path: str):
    return JSONResponse({"ok": True})

@api.get("/api/categories")
async def get_categories(request: Request, session: AsyncSession = Depends(api_session)):
    init_data = get_init_data(request)
    try:
        verify_telegram_init_data(init_data, BOT_TOKEN)
    except Exception:
        raise HTTPException(401, "Bad initData")
    return {"ok": True, "categories": await FoodsService.list_categories(session)}

@api.get("/api/foods")
async def get_foods(request: Request, session: AsyncSession = Depends(api_session)):
    init_data = get_init_data(request)
    try:
        verify_telegram_init_data(init_data, BOT_TOKEN)
    except Exception:
        raise HTTPException(401, "Bad initData")
    return {"ok": True, "foods": await FoodsService.list_foods(session)}

@api.get("/api/promo/validate")
async def validate_promo(code: str, request: Request, session: AsyncSession = Depends(api_session)):
    init_data = get_init_data(request)
    try:
        verify_telegram_init_data(init_data, BOT_TOKEN)
    except Exception:
        raise HTTPException(401, "Bad initData")
    return await PromoService.validate_code(session, code)

# ---------------------------
# App runner: bot + api in one process
# ---------------------------
async def run_api():
    config = uvicorn.Config(api, host=API_HOST, port=API_PORT, log_level=LOG_LEVEL.lower(), loop="asyncio")
    server = uvicorn.Server(config)
    await server.serve()

async def run_bot():
    storage = RedisStorage.from_url(REDIS_URL)
    dp = Dispatcher(storage=storage)
    dp.include_router(client_router)
    dp.include_router(admin_router)
    dp.include_router(courier_router)

    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)

async def main():
    await init_db()
    await asyncio.gather(run_api(), run_bot())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
