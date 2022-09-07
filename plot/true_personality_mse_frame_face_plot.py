import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['AmbFac', 'SENet', 'HRNet', 'Swin', "3DRes", "SlowFast", "TPN", "VAT", "Average"]
frame_acc = [0.9902, 1.14, 1.27, 1.27, 1.18, 1.27, 1.06, 1.02, 1.15]
face_acc = [1.0704, 1.15,  1.28, 1.08, 1.10, 1.33, 1.06, 0.96, 1.13]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(x - width / 2, frame_acc, width, label='Full frame')
face = ax.bar(x + width / 2, face_acc, width, label='Face frame')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE', fontsize=15)
# ax.set_title('True personality MSE scores by frame and face images')
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
plt.ylim(0.9, 1.4)
# plt.xlim(0, 10)
plt.savefig("True_personality_MSE.png", dpi=300)
plt.show()
