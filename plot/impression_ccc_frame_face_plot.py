import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['SENet', 'HRNet', 'Swin', "3DRes", "SlowFast", "TPN", "VAT", "Average"]
frame_ccc = [51.22, 52.70, 23.33, 53.61, 1.9, 24.94, 58.02, 37.96]
face_ccc = [53.79, 61.48, 20.69, 31.85, 2.1, 44.2, 62.98, 39.58]

x = np.arange(len(labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(x - width / 2, frame_ccc, width, label='Full frame')
face = ax.bar(x + width / 2, face_ccc, width, label='Face frame')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CCC (%)', fontsize=15)
# ax.set_title('Impression CCC scores by frame and face images')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel_label(rects, xytxt=(-2, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=xytxt,  # 3 points vertical offset
                    textcoords="offset points",
                    fontsize=8,
                    ha='center', va='bottom')


# def autolabel_face(rects, xytxt=(1, 2)):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=xytxt,  # 3 points vertical offset
#                     textcoords="offset points",
#                     fontsize=8,
#                     ha='center', va='bottom')


autolabel_label(frame)
autolabel_label(face, (4, 4))

fig.tight_layout()
plt.ylim(0, 78)
plt.savefig("Impression_CCC.png", dpi=300)
plt.show()
