"""District-to-licensee lookup carried over unchanged from Capstone I (notebook cell 3-FIX-A).

ERA does not publish official concession polygons; this is an analytical approximation.
"""

DIST_LICENSEE = {
    # ── UMEME: urban/peri-urban networks nationwide ──
    'KAMPALA':'UMEME','WAKISO':'UMEME','MUKONO':'UMEME','BUIKWE':'UMEME',
    'JINJA':'UMEME','IGANGA':'UMEME','MAYUGE':'UMEME','KAMULI':'UMEME',
    'KALIRO':'UMEME','LUUKA':'UMEME','NAMUTUMBA':'UMEME',
    'MBALE':'UMEME','MANAFWA':'UMEME','BUDUDA':'UMEME','SIRONKO':'UMEME',
    'BULAMBULI':'UMEME','TORORO':'UMEME','BUSIA':'UMEME','BUGIRI':'UMEME',
    'NAMAYINGO':'UMEME',
    'GULU':'UMEME','NWOYA':'UMEME','OMORO':'UMEME','KITGUM':'UMEME',
    'PADER':'UMEME','LIRA':'UMEME','ALEBTONG':'UMEME',
    'MBARARA':'UMEME','ISINGIRO':'UMEME','KIRUHURA':'UMEME',
    'MASAKA':'UMEME','KALUNGU':'UMEME','LWENGO':'UMEME','NTUNGAMO':'UMEME',
    'KABAROLE':'UMEME','KASESE':'UMEME',
    'MOROTO':'UMEME','KOTIDO':'UMEME','NAKAPIRIPIRIT':'UMEME','AMUDAT':'UMEME',
    'SOROTI':'UMEME',
    # Additional UMEME urban towns and sub-county names in GIS
    'BOMBO':'UMEME','MITYANA':'UMEME','KABALE':'UMEME','FORTPORTAL':'UMEME',
    'WANDEGEYA':'UMEME','KAYUNGA':'UMEME','KASAMBIRA':'UMEME',
    'KAZO':'UMEME','ENTEBBE':'UMEME','NATETE':'UMEME',
    'NAKULABYE':'UMEME','NAKULABYER':'UMEME',
    'LUGAZI':'UMEME','BANDA':'UMEME','NAJJA':'UMEME','MPIGI':'UMEME',
    'KABALAGALA':'UMEME','KITINTALE':'UMEME','NABONGO':'UMEME',
    'NAMISINDWA':'UMEME','NAKASEKE':'UMEME','KAPCHORWA':'UMEME',
    'LUWERO':'UMEME','NYENDO':'UMEME','MUTUNDWE':'UMEME',
    'KISSITA':'UMEME',

    # ── CNST: Central North Service Territory ──
    'KOLE':'CNST','KWANIA':'CNST','OYAM':'CNST','APAC':'CNST',
    'DOKOLO':'CNST','AMOLATAR':'CNST','MARUZI':'CNST',
    'ALOI':'CNST','OTWAL':'CNST','PABBO':'CNST','KAMDINI':'CNST',

    # ── EST: Eastern Service Territory ──
    'BUDAKA':'EST','KIBUKU':'EST','PALLISA':'EST','BUTEBO':'EST',

    # ── MWST: Mid West Service Territory ──
    'HOIMA':'MWST','MASINDI':'MWST','KIBOGA':'MWST','KYANKWANZI':'MWST',
    'KAKUMIRO':'MWST','KAGADI':'MWST','KIRYANDONGO':'MWST','BULIISA':'MWST',
    'PAKWACH':'MWST','KABWOYA':'MWST','KABWOOYA':'MWST','BIISO':'MWST',
    'KISIITA':'MWST','IGAYAZA':'MWST','KAGAD':'MWST','KAGSDI':'MWST',

    # ── NEST: North East Service Territory ──
    'KUMI':'NEST','BUKEDEA':'NEST','NGORA':'NEST','SERERE':'NEST',
    'AMURIA':'NEST','KATAKWI':'NEST','KABERAMAIDO':'NEST','KAPELEBYONG':'NEST',

    # ── NNWST: North North West Service Territory ──
    'ARUA':'NNWST','NEBBI':'NNWST','ZOMBO':'NNWST','MARACHA':'NNWST',
    'KOBOKO':'NNWST','YUMBE':'NNWST','ADJUMANI':'NNWST','MOYO':'NNWST',
    'MADIOKOLLO':'NNWST','EASTMOYO':'NNWST','ARUAMUNICIPALITY':'NNWST','AYIVU':'NNWST',

    # ── NWST: North Western Service Territory (Kibaale-Kyenjojo) ──
    'KYENJOJO':'NWST','KIBAALE':'NWST','KAMWENGE':'NWST',
    'KARUGUZA':'NWST','KIGANDA':'NWST','KIGAGA':'NWST','KYAKATEMBA':'NWST',

    # ── SST: Southern Service Territory (Masaka-Rakai) ──
    'RAKAI':'SST','LYANTONDE':'SST','SEMBABULE':'SST',
    'MUBENDE':'SST','KASSANDA':'SST',
    'LWEMIYAGA':'SST','RUGAGA':'SST','RUGAAGA':'SST','SANJE':'SST',

    # ── SWST: Southern West Service Territory (Rukungiri-Kanungu) ──
    'RUKUNGIRI':'SWST','KANUNGU':'SWST','BUHWEJU':'SWST',
    'MITOOMA':'SWST','SHEEMA':'SWST','BUSHENYI':'SWST','IBANDA':'SWST',
    'KIHIIHI':'SWST','KIHIHI':'SWST','MUHANGA':'SWST','MUKO':'SWST',
    'KIKAGATI':'SWST','KAKAGATI':'SWST','KIKAGANTI':'SWST','KIKAGATE':'SWST',
    'KISORO':'SWST','LUMBUGU':'SWST',

    # ── BECS: Bundibugyo Energy Cooperative ──
    'BUNDIBUGYO':'BECS','KARUGUTU':'BECS',

    # ── KIL: Kilembe Investment Limited ──
    # KIL operates within Kasese district only around Kilembe mines concession.
    # KITWE is a trading centre adjacent to the Kilembe mines.
    'KITWE':'KIL',

    # ── KRECS: Kyegegwa Rural Electricity Cooperation Society ──
    'KYEGEGWA':'KRECS',

    # ── PACMECS: Pader Abim Cooperative ──
    'ABIM':'PACMECS','AGAGO':'PACMECS',

    # ── Exclude: GIS data quality labels, not real districts ──
    'EXISTING':'EXCLUDE','OPERATIONAL':'EXCLUDE','TBC':'EXCLUDE',
}


def norm(s):
    import pandas as pd
    if pd.isna(s):
        return ""
    return str(s).upper().replace(" ", "").replace("-", "").replace("'", "").replace(".", "")


def lookup_licensee(d_norm):
    if d_norm in ("", "EXISTING", "OPERATIONAL", "TBC"):
        return "EXCLUDE"
    if d_norm in DIST_LICENSEE:
        return DIST_LICENSEE[d_norm]
    for key, val in DIST_LICENSEE.items():
        if len(key) >= 5 and key in d_norm:
            return val
    return "UNMAPPED"
