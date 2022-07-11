import matplotlib.pyplot as plt
import numpy as np


labels = ["IntImg", "PersE", 'SENet', 'HRNet', 'SwTran', "3DRes", "SlowFast", "TPN", "VAT", "Average"]
frame_acc = [10.59, 0.68, 11.91, 10.07, 5.92, 0.17, 0.32, 3.52, 4.38, 5.28]
face_acc = [17.38, 0.45, 15.1,  17.52, 2.56, 0.72, 0.16, 1.25, 0.31, 6.16]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(x - width / 2, frame_acc, width, label='Frame')
face = ax.bar(x + width / 2, face_acc, width, label='Face')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CCC (%)')
ax.set_title('True personality CCC scores by frame and face images')
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
plt.ylim(0, 19)
# plt.xlim(0, 10)
plt.savefig("True_personality_CCC.png", dpi=300)
plt.show()
