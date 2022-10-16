import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['Ope', 'Con', 'Ext', "Agr", "Neu", "Avg"]
frame_ccc = [00.39, 5.36,	1.83,	3.06,	1.16,	2.36]  # audio
segment_ccc = [3.24, 11.13, 2.10, 4.60,	6.52, 5.51]
video_ccc = [0.26,	9.38,	3.12,	5.30,	3.43,	4.30]


x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(2 * x - 0.5, frame_ccc, width, label='Audio')
face = ax.bar(2 * x, segment_ccc, width, label='Visual')
face2 = ax.bar(2 * x + 0.5, video_ccc, width, label='Audiovisual')

x_line = np.arange(-1, 12, 2)
line = ax.plot(x_line, [0] * len(x_line), color="black", linewidth=1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CCC (%)', fontsize=15)
# ax.spines['left'].set_position(("data", 0.5))
# ax.spines['bottom'].set_position(("data", 1))
# ax.set_title('Impression CCC scores by frame and face images')
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(labels)
ax.legend()


def autolabel_label(rects, xytxt=(-2, 2)):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        string = '{:.2f}'.format(height)
        if height < 0:
            height = height - 1.6
        ax.annotate(string,
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


autolabel_label(frame, (-4, 1))
autolabel_label(face, (2, 6))
autolabel_label(face2, (8, 4))


fig.tight_layout()
plt.ylim(0, 12)
plt.savefig("TruePersonality_audio_visual_audiovisual_CCC.png", dpi=300)
plt.show()
