import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['SENet', 'HRNet', 'SwinTrans', "3DResNet", "SlowFast", "TPN", "VAT"]
frame_acc = [90.51, 90.50, 89.07, 90.46, 86.09, 89.18, 90.63]
face_acc = [90.75, 91.13, 89.09,  89.48, 86.50, 90.03, 91.18]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(x - width / 2, frame_acc, width, label='Frame')
face = ax.bar(x + width / 2, face_acc, width, label='Face')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('ACC (%)')
ax.set_title('Impression ACC scores by frame and face images')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel_frame(rects, xytxt=(-1, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


def autolabel_face(rects, xytxt=(1, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


autolabel_frame(frame)
autolabel_face(face)

fig.tight_layout()
plt.ylim(70, 100)
plt.savefig("Impression_ACC.png", dpi=300)
plt.show()
