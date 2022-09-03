import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['ResNet', 'SENet', 'HRNet', 'SwTran', "3DRes", "SlowFast", "TPN", "VAT", "Average"]
frame_acc = [91.01, 90.51, 90.50, 89.07, 90.46, 86.09, 89.18, 90.63, 89.68]
face_acc = [90.75, 90.75, 91.13, 89.09,  89.48, 86.50, 90.03, 91.18, 89.86]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(x - width / 2, frame_acc, width, label='Full frame')
face = ax.bar(x + width / 2, face_acc, width, label='Face frame')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ACC (%)', fontsize=15)
# ax.set_title('Impression ACC scores by frame and face images')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel_frame(rects, xytxt=(-2, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


def autolabel_face(rects, xytxt=(2, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


autolabel_frame(frame)
autolabel_face(face)

fig.tight_layout()
plt.ylim(85, 93)
plt.savefig("Impression_ACC.png", dpi=300)
plt.show()
