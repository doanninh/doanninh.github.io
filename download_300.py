import zipfile
import random
import shutil
from pathlib import Path


# ============================================================
# CẤU HÌNH
# ============================================================

BASE_DIR = Path(r"D:\plant_ai_project")

ZIP_DIR = BASE_DIR / "dataset_download"
OUTPUT_DIR = BASE_DIR / "dataset_300"

TARGET = 100

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# TÌM FILE ZIP
# ============================================================

def find_zip_file():

    zip_files = list(ZIP_DIR.glob("*.zip"))

    if not zip_files:
        print("KHÔNG TÌM THẤY FILE ZIP!")
        print()
        print("Hãy kiểm tra thư mục:")
        print(ZIP_DIR)
        return None

    print("Đã tìm thấy file ZIP:")

    for file in zip_files:
        print("  ", file.name)

    return zip_files[0]


# ============================================================
# XÁC ĐỊNH CLASS CỦA ẢNH
# ============================================================

def get_class_from_path(path_inside_zip):

    parts = [
        part.lower()
        for part in Path(path_inside_zip).parts
    ]

    if "healthy" in parts:
        return "healthy"

    if "wilted" in parts:
        return "wilted"

    if "mildew" in parts:
        return "diseased"

    return None


# ============================================================
# LẤY DANH SÁCH ẢNH TRONG ZIP
# ============================================================

def collect_images(zip_file):

    classes = {
        "healthy": [],
        "wilted": [],
        "diseased": []
    }

    with zipfile.ZipFile(
        zip_file,
        "r"
    ) as z:

        for item in z.infolist():

            if item.is_dir():
                continue

            path = item.filename

            extension = Path(path).suffix.lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            class_name = get_class_from_path(path)

            if class_name is None:
                continue

            classes[class_name].append(item)

    return classes


# ============================================================
# TẠO THƯ MỤC
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():

        print()
        print("Đang xóa dataset_300 cũ...")

        shutil.rmtree(OUTPUT_DIR)

    for class_name in [
        "healthy",
        "wilted",
        "diseased"
    ]:

        folder = OUTPUT_DIR / class_name

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COPY ẢNH TỪ ZIP
# ============================================================

def extract_images(
    zip_file,
    image_list,
    class_name
):

    print()
    print(
        f"{class_name.upper()}: "
        f"tìm thấy {len(image_list)} ảnh"
    )

    if len(image_list) < TARGET:

        print(
            f"KHÔNG ĐỦ 100 ẢNH cho {class_name}!"
        )

        return False

    # Trộn ngẫu nhiên
    random.shuffle(image_list)

    selected = image_list[:TARGET]

    destination_folder = (
        OUTPUT_DIR / class_name
    )

    with zipfile.ZipFile(
        zip_file,
        "r"
    ) as z:

        for index, item in enumerate(
            selected,
            start=1
        ):

            extension = Path(
                item.filename
            ).suffix.lower()

            output_file = (
                destination_folder
                / f"{class_name}_{index:03d}{extension}"
            )

            with z.open(item) as source:

                with open(
                    output_file,
                    "wb"
                ) as destination:

                    shutil.copyfileobj(
                        source,
                        destination
                    )

            print(
                f"\rĐã lấy: {index}/100",
                end=""
            )

    print()

    return True


# ============================================================
# ĐẾM ẢNH
# ============================================================

def count_images(folder):

    return len([
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


# ============================================================
# KIỂM TRA
# ============================================================

def check_result():

    print()
    print("=" * 60)
    print("KIỂM TRA DATASET")
    print("=" * 60)

    healthy = count_images(
        OUTPUT_DIR / "healthy"
    )

    wilted = count_images(
        OUTPUT_DIR / "wilted"
    )

    diseased = count_images(
        OUTPUT_DIR / "diseased"
    )

    total = (
        healthy
        + wilted
        + diseased
    )

    print()
    print(
        f"healthy  : {healthy}/100"
    )

    print(
        f"wilted   : {wilted}/100"
    )

    print(
        f"diseased : {diseased}/100"
    )

    print(
        f"TỔNG     : {total}/300"
    )

    print()

    if (
        healthy == 100
        and wilted == 100
        and diseased == 100
    ):

        print("========================================")
        print("✓ DATASET ĐÃ ĐỦ 300 ẢNH")
        print("========================================")
        print()
        print("Dataset nằm tại:")
        print(OUTPUT_DIR)

    else:

        print("✗ DATASET CHƯA ĐỦ 300 ẢNH")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("       TẠO DATASET 300 ẢNH")
    print("=" * 60)

    # Giữ kết quả ngẫu nhiên ổn định
    random.seed(42)

    # --------------------------------------------------------
    # TÌM ZIP
    # --------------------------------------------------------

    zip_file = find_zip_file()

    if zip_file is None:
        return

    # --------------------------------------------------------
    # ĐỌC ZIP
    # --------------------------------------------------------

    print()
    print("Đang kiểm tra nội dung ZIP...")

    classes = collect_images(
        zip_file
    )

    print()
    print("Kết quả tìm thấy:")

    print(
        "Healthy :",
        len(classes["healthy"])
    )

    print(
        "Wilted  :",
        len(classes["wilted"])
    )

    print(
        "Mildew  :",
        len(classes["diseased"])
    )

    # --------------------------------------------------------
    # KIỂM TRA ĐỦ 100 MỖI LỚP
    # --------------------------------------------------------

    if len(classes["healthy"]) < 100:

        print("Không đủ ảnh healthy.")
        return

    if len(classes["wilted"]) < 100:

        print("Không đủ ảnh wilted.")
        return

    if len(classes["diseased"]) < 100:

        print("Không đủ ảnh mildew/diseased.")
        return

    # --------------------------------------------------------
    # TẠO DATASET
    # --------------------------------------------------------

    prepare_output()

    # --------------------------------------------------------
    # LẤY 100 HEALTHY
    # --------------------------------------------------------

    extract_images(
        zip_file,
        classes["healthy"],
        "healthy"
    )

    # --------------------------------------------------------
    # LẤY 100 WILTED
    # --------------------------------------------------------

    extract_images(
        zip_file,
        classes["wilted"],
        "wilted"
    )

    # --------------------------------------------------------
    # LẤY 100 DISEASED
    # --------------------------------------------------------

    extract_images(
        zip_file,
        classes["diseased"],
        "diseased"
    )

    # --------------------------------------------------------
    # KIỂM TRA
    # --------------------------------------------------------

    check_result()


# ============================================================
# CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":
    main()