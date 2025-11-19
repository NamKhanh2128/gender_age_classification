from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 

def plot_probabilities(classes, probabilities, title, frame):
    import numpy as np
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt

    # Xóa biểu đồ cũ nếu có
    for widget in frame.winfo_children():
        widget.destroy()

    # Tăng khoảng cách giữa các cột bằng cách nhân chỉ số x
    x = np.arange(len(classes))  # Tạo các vị trí cột
    spacing = 1.2                # Điều chỉnh khoảng cách giữa các nhãn (1.0 là mặc định)

    x_spaced = x * spacing

    # Tạo biểu đồ
    fig = plt.Figure(figsize=(len(classes) * 0.9, 4), dpi=100)
    ax = fig.add_subplot(111)

    bars = ax.bar(x_spaced, probabilities, width=0.5, color="#4A90E2", align='center')

    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Xác suất")

    # Đặt nhãn trục X cân đối với cột
    ax.set_xticks(x_spaced)
    ax.set_xticklabels(classes, rotation=30, ha='center')

    # Hiển thị % xác suất trên đầu mỗi cột
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.02,
                f"{probabilities[i]:.1%}", ha='center', va='bottom', fontsize=8)

    # Nhúng biểu đồ vào Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()