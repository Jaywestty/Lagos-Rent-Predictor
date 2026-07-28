import json
from pathlib import Path

LGA_LCDA_MAP = {
    "Alimosho": ["Agbado Oke-Odo", "Ayobo Ipaja", "Egbe Idimu", "Ikotun Igando", "Mosan Okunola"],
    "Ikeja": ["Ojodu", "Onigbongbo"],
    "Kosofe": ["Ikosi Isheri", "Agboyi Ketu"],
    "Eti-Osa": ["Iru Victoria Island", "Ikoyi Obalende", "Eti-Osa East"],
    "Ifako-Ijaiye": ["Ifako Ogba", "Ojokoro"],
    "Agege": ["Orile Agege"],
    "Apapa": ["Apapa Iganmu"],
    "Mushin": ["Odi Olowo Ojuwoye"],
    "Oshodi-Isolo": ["Isolo", "Ejigbo"],
    "Shomolu": ["Bariga"],
    "Surulere": ["Itire Ikate", "Coker Aguda"],
    "Badagry": ["Badagry West", "Olorunda"],
    "Lagos Island": ["Lagos Island East"],
    "Ikorodu": ["Ikorodu West", "Imota", "Igbogbo Baiyeku", "Ijede"],
    "Epe": ["Eredo", "Ikosi Ejinrin"],
    "Ibeju-Lekki": ["Lekki"],
    "Ojo": ["Iba"],
    "Ajeromi-Ifelodun": [],
    "Amuwo-Odofin": [],
    "Lagos Mainland": [],
}


def generate_suffix_variants(lga, lcda):
    lga_words = lga.replace("-", " ").split()
    lcda_words = lcda.replace("-", " ").split()

    variants = set()
    variants.add(f"{lga} {lcda}")
    variants.add(" ".join(lga_words + lcda_words))
    variants.add(" ".join(lga_words + [lcda_words[0]]))
    if len(lga_words) > 1:
        variants.add(" ".join([lga_words[0]] + lcda_words))
        variants.add(" ".join([lga_words[0], lcda_words[0]]))

    return variants


def main():
    suffixes = set()
    for lga, lcdas in LGA_LCDA_MAP.items():
        suffixes.add(lga.lower())
        for lcda in lcdas:
            suffixes.add(lcda.lower())
            for variant in generate_suffix_variants(lga, lcda):
                suffixes.add(variant.lower())

    out_path = Path("data/known_admin_suffixes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(suffixes), f, indent=2)

    print(f"Generated {len(suffixes)} admin suffix variants to {out_path}")


if __name__ == "__main__":
    main()