"""Single source of truth for the fictional Hearthwise world.

Every dataset is derived from these tables so that the knowledge base, the
evaluation questions, the tickets and the offline web never contradict each
other by accident. Contradictions that exist on purpose (traps) are created
explicitly in the generators and documented in datasets/README.md.

All companies, products, organisations, people and numbers here are fictional.
"""

from __future__ import annotations

SEED = 42

COMPANY = {
    "name": "Hearthwise",
    "domain": "hearthwise.example",
    "support_email": "support@hearthwise.example",
    "hq": "Valencia, Spain",
    "employees": 350,
}

# --------------------------------------------------------------------------- #
# Hearthwise products                                                         #
# --------------------------------------------------------------------------- #

PRODUCTS = [
    {
        "sku": "HW-EMB-T3",
        "name": "Ember T3",
        "slug": "ember-t3",
        "category": "smart thermostat",
        "price_eur": 149,
        "launched": "2023-10-02",
        "wifi": "2.4 GHz",
        "standby_w": 0.6,
        "power": "24 V wired (C-wire) with 2 x AA battery backup",
        "needs_hub": False,
        "highlights": [
            "3.5-inch colour touch display",
            "learns your schedule in about 7 days",
            "geofencing: lowers the temperature when everyone leaves",
            "works with HomeMesh 2.0 through the Hearth Hub",
        ],
        "reset": "Press and hold the dial for 15 seconds until the screen shows RESET, then release.",
    },
    {
        "sku": "HW-GLW-A19",
        "name": "Glow A19",
        "slug": "glow-a19",
        "category": "smart bulb",
        "price_eur": 12,
        "launched": "2022-03-14",
        "wifi": "2.4 GHz",
        "standby_w": 0.4,
        "power": "9 W LED, E27 base",
        "needs_hub": False,
        "lumens": 806,
        "highlights": [
            "806 lumens (equivalent to a 60 W incandescent bulb)",
            "tunable white from 2700 K to 6500 K plus 16 million colours",
            "rated life of 25,000 hours",
        ],
        "reset": "Switch the bulb off and on five times in a row; it blinks twice when reset.",
    },
    {
        "sku": "HW-SPK-MINI",
        "name": "Spark Mini",
        "slug": "spark-mini",
        "category": "smart plug",
        "price_eur": 19,
        "launched": "2022-03-14",
        "wifi": "2.4 GHz",
        "standby_w": 0.3,
        "power": "max load 3680 W (16 A at 230 V)",
        "needs_hub": False,
        "highlights": [
            "real-time energy monitoring in the app",
            "max load 3680 W (16 A at 230 V)",
            "schedules and away mode that randomises lights",
        ],
        "reset": "Hold the side button for 10 seconds until the LED blinks red quickly.",
    },
    {
        "sku": "HW-WCH-S1",
        "name": "Watcher S1",
        "slug": "watcher-s1",
        "category": "motion sensor",
        "price_eur": 29,
        "launched": "2023-04-03",
        "wifi": "none (HomeMesh radio)",
        "standby_w": 0.0,
        "power": "CR2450 coin battery, about 18 months of life",
        "needs_hub": True,
        "highlights": [
            "7 m detection range with a 120-degree field of view",
            "requires the Hearth Hub (it talks HomeMesh, not Wi-Fi)",
            "battery lasts about 18 months with default settings",
        ],
        "reset": "Insert a paper clip into the pinhole for 5 seconds until the LED flashes blue.",
    },
    {
        "sku": "HW-LKT-C2",
        "name": "Lookout C2",
        "slug": "lookout-c2",
        "category": "indoor camera",
        "price_eur": 89,
        "launched": "2025-09-01",
        "wifi": "2.4 GHz and 5 GHz",
        "standby_w": 1.8,
        "power": "USB-C, 5 V / 1 A adapter included",
        "needs_hub": False,
        "highlights": [
            "2K video with night vision up to 8 m",
            "local recording to a microSD card of up to 256 GB",
            "cloud video history requires Hearthwise Plus",
            "first Hearthwise device that supports 5 GHz Wi-Fi",
        ],
        "reset": "Hold the reset button under the base for 10 seconds until you hear a chime.",
    },
    {
        "sku": "HW-HUB-2",
        "name": "Hearth Hub",
        "slug": "hearth-hub",
        "category": "smart home hub",
        "price_eur": 69,
        "launched": "2023-04-03",
        "wifi": "2.4 GHz plus Ethernet port",
        "standby_w": 1.2,
        "power": "USB-C, 5 V / 2 A adapter included",
        "needs_hub": False,
        "max_devices": 64,
        "highlights": [
            "connects up to 64 HomeMesh devices",
            "runs automations locally, even when the internet is down",
            "Ethernet port for a more stable connection",
        ],
        "reset": "Hold the button on the back for 20 seconds until the ring light turns orange.",
    },
]

PLUS = {
    "name": "Hearthwise Plus",
    "monthly_eur": 4.99,
    "yearly_eur": 49,
    "video_history_days": 30,
    "warranty_extension_years": 1,
}

# --------------------------------------------------------------------------- #
# Policies (current + one superseded version used as a trap)                 #
# --------------------------------------------------------------------------- #

RETURNS_CURRENT = {
    "version": "2025",
    "effective": "2025-03-01",
    "window_days": 30,
    "restocking_fee": "none",
    "refund_days": "5 to 7 business days",
}
RETURNS_OLD = {
    "version": "2023",
    "effective": "2023-01-15",
    "superseded": "2025-03-01",
    "window_days": 14,
    "restocking_fee": "10% for opened items",
    "refund_days": "10 business days",
}
WARRANTY = {"years": 2, "plus_years": 3}
SHIPPING = {
    "free_threshold_eur": 49,
    "standard_fee_eur": 4.95,
    "standard_days": "2 to 4 business days",
    "express_fee_eur": 6.95,
    "express_days": "1 business day",
    "eu_fee_eur": 7.95,
    "eu_days": "3 to 6 business days",
}
# Only documented in Spanish (kb-031): a cross-lingual retrieval trap.
LATAM_SHIPPING = {
    "fee_eur": 14.95,
    "countries": {"México": "7 a 10", "Colombia": "8 a 12", "Chile": "9 a 12"},
    "customs": "los aranceles y tasas de aduana los paga la persona que recibe el pedido",
}

INTENTS = [
    "order_status",
    "return_request",
    "warranty_claim",
    "connectivity_issue",
    "billing_question",
    "account_access",
    "product_question",
    "cancel_subscription",
]

# --------------------------------------------------------------------------- #
# The offline web: fictional facts (some with deliberate contradictions)     #
# --------------------------------------------------------------------------- #

COUNTRIES = ["Spain", "Portugal", "Mexico", "Colombia", "Chile"]

MARKET_2025 = {  # source: marketlens.example (credible, fictional)
    "Spain": {"thermostat_units": 410_000, "adoption_pct": 21, "growth_pct": 6},
    "Portugal": {"thermostat_units": 95_000, "adoption_pct": 17, "growth_pct": 9},
    "Mexico": {"thermostat_units": 260_000, "adoption_pct": 14, "growth_pct": 18},
    "Colombia": {"thermostat_units": 120_000, "adoption_pct": 9, "growth_pct": 22},
    "Chile": {"thermostat_units": 85_000, "adoption_pct": 16, "growth_pct": 12},
}
ELECTRICITY_EUR_KWH_2025 = {"Spain": 0.19, "Portugal": 0.21, "Mexico": 0.09, "Colombia": 0.17, "Chile": 0.22}
DELIVERY_DAYS_2025 = {"Mexico": 6, "Colombia": 8, "Chile": 7}

COMPETITORS = {
    "corvid_thermo_x": {
        "brand": "Corvid Home",
        "name": "Corvid Thermo X",
        "price_eur": 179,
        "launched": "2023-05-22",
        "standby_w_old": 0.9,  # review from 2024-02-08
        "standby_w_new": 0.4,  # updated review after firmware 3.1, 2025-06-12
        "sold_in": ["Spain", "Portugal", "Mexico", "Chile"],
    },
    "brightline_bulb_pro": {
        "brand": "Brightline",
        "name": "Brightline Bulb Pro",
        "price_eur": 18,
        "lumens": 1100,
        "sold_in": ["Spain", "Portugal", "Mexico", "Colombia", "Chile"],
    },
    "tessera_cam_4": {
        "brand": "Tessera",
        "name": "Tessera Cam 4",
        "price_eur": 99,
        "resolution": "1080p",
        "sold_in": ["Spain", "Portugal", "Colombia"],
    },
}

HOMEMESH = {"v1_max_devices": 100, "v2_max_devices": 250, "v2_released": "2024-04-16"}
EHEO = {  # fictional European Home Energy Observatory
    "standby_recommended_max_w": 0.5,
    "savings_low_pct": 8,
    "savings_high_pct": 12,
    "study_homes": 1_240,
    "study_published": "2025-02-20",
}
