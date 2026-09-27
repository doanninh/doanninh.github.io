import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# =========================================================
# CẤU HÌNH
# =========================================================
PROJECT_DIR = r"D:\plant_ai_project"

MODEL_PATH = os.path.join(PROJECT_DIR, "plant_model.keras")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "class_names.json")

IMG_SIZE = (224, 224)

# Ngưỡng để tránh app kết luận quá tự tin khi model còn phân vân
CONFIDENCE_THRESHOLD = 0.75
MARGIN_THRESHOLD = 0.15

CLASS_NAMES_DEFAULT = ["diseased", "healthy", "wilted"]

DISPLAY_NAMES = {
    "healthy": "🌿 CÂY KHỎE MẠNH",
    "wilted": "🥀 CÂY ĐANG HÉO",
    "diseased": "🦠 CÂY CÓ DẤU HIỆU BỆNH"
}

CARE_SUGGESTIONS = {
    "healthy": (
        "Cây đang ở trạng thái tốt.\n\n"
        "• Tiếp tục duy trì ánh sáng phù hợp.\n"
        "• Tưới nước vừa đủ, tránh úng.\n"
        "• Theo dõi lá và thân thường xuyên.\n"
        "• Duy trì dinh dưỡng phù hợp."
    ),
    "wilted": (
        "⚠ Tình trạng cây không tốt.\n\n"
        "• Kiểm tra độ ẩm của đất.\n"
        "• Kiểm tra tình trạng rễ.\n"
        "• Tránh để cây thiếu nước quá lâu.\n"
        "• Điều chỉnh ánh sáng nếu cây bị nắng quá mạnh."
    ),
    "diseased": (
        "⚠ Cây có dấu hiệu không khỏe.\n\n"
        "• Kiểm tra lá, thân và rễ.\n"
        "• Kiểm tra sâu bệnh hoặc nấm.\n"
        "• Cân nhắc cách ly cây nghi bị bệnh.\n"
        "• Không tưới quá nhiều nước."
    )
}

# =========================================================
# KIỂM TRA FILE
# =========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy model:\n{MODEL_PATH}\n\n"
        "Hãy chạy train.py trước."
    )

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    class_names = CLASS_NAMES_DEFAULT

if class_names != CLASS_NAMES_DEFAULT:
    print("CẢNH BÁO: class_names.json khác mapping chuẩn.")
    print("Đang sử dụng mapping trong class_names.json:")
    print(class_names)

print("Đang tải model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Đã tải model.")

# =========================================================
# BIẾN GIAO DIỆN
# =========================================================
root = tk.Tk()
root.title("NHẬN DIỆN VÀ PHÂN LOẠI TÌNH TRẠNG CÂY")
root.geometry("1100x760")
root.minsize(950, 680)

selected_image_path = None
history = []

# =========================================================
# HÀM TIỆN ÍCH
# =========================================================
def make_prediction(image_path):
    """
    Model đã chứa preprocess_input bên trong train.py,
    vì vậy app KHÔNG preprocess lần thứ hai.
    """

    image = Image.open(image_path).convert("RGB")
    original_for_display = image.copy()

    image_resized = image.resize(IMG_SIZE)

    img_array = np.array(image_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán ảnh gốc
    prediction_original = model.predict(
        img_array,
        verbose=0
    )[0]

    # Dự đoán thêm ảnh lật ngang để tăng độ ổn định
    flipped_image = image_resized.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    flipped_array = np.array(
        flipped_image,
        dtype=np.float32
    )

    flipped_array = np.expand_dims(
        flipped_array,
        axis=0
    )

    prediction_flipped = model.predict(
        flipped_array,
        verbose=0
    )[0]

    # Trung bình hai góc nhìn
    predictions = (
        prediction_original * 0.7
        + prediction_flipped * 0.3
    )

    # Chuẩn hóa lại
    predictions = predictions / np.sum(predictions)

    best_index = int(np.argmax(predictions))
    best_class = class_names[best_index]

    sorted_indices = np.argsort(predictions)[::-1]

    top1 = float(predictions[sorted_indices[0]])
    top2 = float(predictions[sorted_indices[1]])

    margin = top1 - top2

    return (
        original_for_display,
        predictions,
        best_class,
        top1,
        margin,
        sorted_indices
    )


def get_health_assessment(
    best_class,
    confidence,
    margin
):
    """
    Có 2 khái niệm:

    1. confidence:
       điểm softmax của model.

    2. health_score:
       điểm sức khỏe được ứng dụng quy đổi,
       KHÔNG phải xác suất AI.
    """

    confident = (
        confidence >= CONFIDENCE_THRESHOLD
        and margin >= MARGIN_THRESHOLD
    )

    if not confident:
        return {
            "certain": False,
            "score": None,
            "title": "⚠ CHƯA ĐỦ CHẮC CHẮN",
            "message": (
                "Mô hình chưa đủ chắc chắn để kết luận.\n\n"
                f"Độ tin cậy cao nhất: {confidence * 100:.2f}%\n"
                f"Chênh lệch Top 1 - Top 2: {margin * 100:.2f}%\n\n"
                "Hãy thử ảnh rõ hơn, chụp toàn bộ cây "
                "và đảm bảo đủ ánh sáng."
            )
        }

    # -----------------------------------------------------
    # CÂY KHỎE
    # -----------------------------------------------------
    if best_class == "healthy":

        # Từ 80% đến 100%
        score = 80 + confidence * 20
        score = min(100, score)

        return {
            "certain": True,
            "score": score,
            "title": "🌿 ĐÁNH GIÁ TỐT",
            "message": (
                "Cây được đánh giá đang ở trạng thái khỏe mạnh.\n"
                "Có thể tiếp tục chăm sóc theo chế độ hiện tại."
            )
        }

    # -----------------------------------------------------
    # CÂY HÉO / BỆNH
    # -----------------------------------------------------
    if best_class in ("wilted", "diseased"):

        # Điểm sức khỏe thấp, tối đa 39%.
        # Model càng chắc chắn về héo/bệnh thì điểm càng thấp.
        score = 39 * (1 - confidence)
        score = max(5, min(39, score))

        return {
            "certain": True,
            "score": score,
            "title": "⚠ ĐÁNH GIÁ XẤU",
            "message": (
                "Cây có dấu hiệu không khỏe.\n"
                "Nên kiểm tra và xử lý tình trạng của cây sớm."
            )
        }

    return {
        "certain": False,
        "score": None,
        "title": "⚠ CHƯA XÁC ĐỊNH",
        "message": "Không xác định được tình trạng cây."
    }


def update_history(
    file_path,
    best_class,
    confidence,
    health_score
):
    filename = os.path.basename(file_path)

    if health_score is None:
        score_text = "Chưa xác định"
    else:
        score_text = f"{health_score:.1f}%"

    history.append(
        f"{filename} | "
        f"{DISPLAY_NAMES.get(best_class, best_class)} | "
        f"AI: {confidence * 100:.1f}% | "
        f"Sức khỏe: {score_text}"
    )

    history_list.delete(0, tk.END)

    # Hiển thị mới nhất ở trên
    for item in reversed(history[-10:]):
        history_list.insert(tk.END, item)


# =========================================================
# CHỌN ẢNH
# =========================================================
def choose_image():

    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Chọn ảnh cây",
        filetypes=[
            (
                "Image files",
                "*.jpg *.jpeg *.png *.bmp *.webp"
            )
        ]
    )

    if not file_path:
        return

    selected_image_path = file_path

    try:
        image = Image.open(file_path).convert("RGB")

        # Thumbnail
        preview = image.copy()
        preview.thumbnail((500, 360))

        photo = ImageTk.PhotoImage(preview)

        image_label.configure(image=photo)
        image_label.image = photo

        filename_label.config(
            text=os.path.basename(file_path)
        )

        # Reset kết quả
        result_label.config(
            text="Đã chọn ảnh — nhấn NHẬN DIỆN",
            fg="#333333"
        )

        confidence_label.config(
            text="Độ tin cậy AI: --"
        )

        health_score_label.config(
            text="Mức đánh giá sức khỏe: --"
        )

        top3_label.config(
            text="Top 3 dự đoán: --"
        )

        care_label.config(
            text="Hướng dẫn chăm sóc: --"
        )

    except Exception as e:
        messagebox.showerror(
            "Lỗi",
            f"Không thể mở ảnh:\n{e}"
        )


# =========================================================
# NHẬN DIỆN
# =========================================================
def predict_image():

    global selected_image_path

    if not selected_image_path:
        messagebox.showwarning(
            "Chưa chọn ảnh",
            "Hãy chọn một ảnh cây trước."
        )
        return

    try:

        (
            image,
            predictions,
            best_class,
            confidence,
            margin,
            sorted_indices
        ) = make_prediction(selected_image_path)

        # -----------------------------------------------
        # Đánh giá
        # -----------------------------------------------
        assessment = get_health_assessment(
            best_class,
            confidence,
            margin
        )

        # -----------------------------------------------
        # Kết quả AI
        # -----------------------------------------------
        result_label.config(
            text=DISPLAY_NAMES.get(
                best_class,
                best_class
            )
        )

        # -----------------------------------------------
        # Confidence thật của model
        # -----------------------------------------------
        confidence_label.config(
            text=(
                f"Độ tin cậy AI: "
                f"{confidence * 100:.2f}%"
            )
        )

        # -----------------------------------------------
        # Health score
        # -----------------------------------------------
        if assessment["score"] is None:

            health_score_label.config(
                text=(
                    "Mức đánh giá sức khỏe: "
                    "CHƯA ĐỦ DỮ LIỆU"
                )
            )

        else:

            health_score_label.config(
                text=(
                    f"Mức đánh giá sức khỏe: "
                    f"{assessment['score']:.2f}%"
                )
            )

        # -----------------------------------------------
        # Hiển thị nhận xét
        # -----------------------------------------------
        if assessment["certain"]:

            care_label.config(
                text=(
                    f"{assessment['title']}\n\n"
                    f"{assessment['message']}\n\n"
                    f"{CARE_SUGGESTIONS.get(best_class, '')}"
                )
            )

        else:

            care_label.config(
                text=(
                    f"{assessment['title']}\n\n"
                    f"{assessment['message']}"
                )
            )

        # -----------------------------------------------
        # Top 3
        # -----------------------------------------------
        top3_lines = []

        for rank, index in enumerate(
            sorted_indices[:3],
            start=1
        ):

            class_name = class_names[index]
            percentage = predictions[index] * 100

            display_name = DISPLAY_NAMES.get(
                class_name,
                class_name
            )

            top3_lines.append(
                f"{rank}. {display_name}: "
                f"{percentage:.2f}%"
            )

        top3_label.config(
            text="\n".join(top3_lines)
        )

        # -----------------------------------------------
        # Lịch sử
        # -----------------------------------------------
        update_history(
            selected_image_path,
            best_class,
            confidence,
            assessment["score"]
        )

    except Exception as e:

        messagebox.showerror(
            "Lỗi nhận diện",
            str(e)
        )


def clear_result():

    global selected_image_path

    selected_image_path = None

    image_label.configure(image="")
    image_label.image = None

    filename_label.config(
        text="Chưa chọn ảnh"
    )

    result_label.config(
        text="Kết quả sẽ xuất hiện ở đây",
        fg="#333333"
    )

    confidence_label.config(
        text="Độ tin cậy AI: --"
    )

    health_score_label.config(
        text="Mức đánh giá sức khỏe: --"
    )

    top3_label.config(
        text="Top 3 dự đoán: --"
    )

    care_label.config(
        text="Hướng dẫn chăm sóc: --"
    )


# =========================================================
# GIAO DIỆN
# =========================================================
title = tk.Label(
    root,
    text="🌱 NHẬN DIỆN VÀ PHÂN LOẠI TÌNH TRẠNG CÂY",
    font=("Arial", 22, "bold")
)

title.pack(pady=15)

main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

# ---------------- LEFT ----------------
left_frame = tk.Frame(
    main_frame,
    width=530
)

left_frame.pack(
    side="left",
    fill="both",
    expand=True,
    padx=10
)

image_title = tk.Label(
    left_frame,
    text="ẢNH CÂY",
    font=("Arial", 16, "bold")
)

image_title.pack(pady=5)

image_label = tk.Label(
    left_frame,
    text="Chưa chọn ảnh",
    width=50,
    height=20,
    relief="solid"
)

image_label.pack(
    fill="both",
    expand=True,
    pady=10
)

filename_label = tk.Label(
    left_frame,
    text="Chưa chọn ảnh",
    font=("Arial", 10)
)

filename_label.pack(pady=5)

button_frame = tk.Frame(left_frame)
button_frame.pack(pady=10)

choose_button = tk.Button(
    button_frame,
    text="📁 CHỌN ẢNH",
    command=choose_image,
    font=("Arial", 12, "bold"),
    width=16,
    height=2
)

choose_button.pack(
    side="left",
    padx=5
)

predict_button = tk.Button(
    button_frame,
    text="🔍 NHẬN DIỆN",
    command=predict_image,
    font=("Arial", 12, "bold"),
    width=16,
    height=2
)

predict_button.pack(
    side="left",
    padx=5
)

clear_button = tk.Button(
    button_frame,
    text="🗑 XÓA",
    command=clear_result,
    font=("Arial", 12, "bold"),
    width=10,
    height=2
)

clear_button.pack(
    side="left",
    padx=5
)

# ---------------- RIGHT ----------------
right_frame = tk.Frame(main_frame)
right_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10
)

result_title = tk.Label(
    right_frame,
    text="KẾT QUẢ NHẬN DIỆN",
    font=("Arial", 16, "bold")
)

result_title.pack(pady=5)

result_label = tk.Label(
    right_frame,
    text="Kết quả sẽ xuất hiện ở đây",
    font=("Arial", 19, "bold"),
    wraplength=450
)

result_label.pack(pady=12)

confidence_label = tk.Label(
    right_frame,
    text="Độ tin cậy AI: --",
    font=("Arial", 14, "bold")
)

confidence_label.pack(pady=5)

health_score_label = tk.Label(
    right_frame,
    text="Mức đánh giá sức khỏe: --",
    font=("Arial", 16, "bold")
)

health_score_label.pack(pady=8)

# Top 3
top3_title = tk.Label(
    right_frame,
    text="TOP 3 DỰ ĐOÁN",
    font=("Arial", 13, "bold")
)

top3_title.pack(pady=(15, 5))

top3_label = tk.Label(
    right_frame,
    text="--",
    font=("Arial", 11),
    justify="left"
)

top3_label.pack(pady=5)

# Nhận xét
care_title = tk.Label(
    right_frame,
    text="NHẬN XÉT / HƯỚNG DẪN",
    font=("Arial", 13, "bold")
)

care_title.pack(pady=(15, 5))

care_label = tk.Label(
    right_frame,
    text="--",
    font=("Arial", 11),
    justify="left",
    wraplength=470
)

care_label.pack(pady=5)

# ---------------- HISTORY ----------------
history_title = tk.Label(
    root,
    text="LỊCH SỬ NHẬN DIỆN",
    font=("Arial", 13, "bold")
)

history_title.pack(pady=(5, 3))

history_list = tk.Listbox(
    root,
    height=6,
    font=("Consolas", 10)
)

history_list.pack(
    fill="x",
    padx=25,
    pady=(0, 15)
)

root.mainloop()
