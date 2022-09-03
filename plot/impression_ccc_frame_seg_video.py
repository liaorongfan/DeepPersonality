import matplotlib
import matplotlib.pyplot as plt
import numpy as np

""" IMPRESSION
frame-level:

    Interprt-img      & 0.5882 & 0.6550 & 0.6326 & 0.5003 & 0.6199 & 0.5992
    senet             & 0.5300 & 0.5580 & 0.5815 & 0.4493 & 0.5708 & 0.5379
    hrnet             & 0.5923 & 0.6912 & 0.6436 & 0.5195 & 0.6273 & 0.6148
    swin-transformer  & 0.2223 & 0.2426 & 0.2531 & 0.1224 & 0.1942 & 0.2069
    mean              0.4832	 0.5367	  0.5277   0.3979	0.5030	 0.4897  
        
segment-level:

    ResNet            & 0.5109 & 0.6156 & 0.5757 & 0.4391 & 0.5366 & 0.5356
    3d-resnet         & 0.3248 & 0.3601 & 0.3601 & 0.2120 & 0.3352 & 0.3185
    slow-fast         & 0.0256 & 0.0320 & 0.0185 & 0.0105 & 0.0184 & 0.0210
    tpn               & 0.4427 & 0.4767 & 0.4998 & 0.3230 & 0.4675 & 0.4420
    vat               & 0.6216 & 0.6753 & 0.6836 & 0.5228 & 0.6456 & 0.6298
    mean                0.38512	0.43194	0.42754	0.30148	0.40066	0.38938	
        
video-level:

    3d-resnet         & 0.1171 & 0.1207 & 0.1199 & 0.0387 & 0.0954 & 0.0983
    slow-fast         & 0.0318 & 0.0463 & 0.0255 & 0.0176 & 0.0297 & 0.0302
    tpn               & 0.0709 & -0.0188 & 0.0166 & -0.0007 & 0.0654 & 0.0267
    vat               & 0.1525 & 0.1794 & 0.1727 & 0.0783 & 0.1457 & 0.1457
    mean              0.093075	0.0819	0.083675	0.033475	0.08405	0.075225	
         
"""


labels = ['O', 'C', 'E', "A", "N", "Avg"]
frame_ccc = [48.32, 53.67, 52.77, 39.79, 50.30, 48.97]
segment_ccc = [38.51, 43.19, 42.75,	30.15, 40.07, 38.94]
face2_ccc = [9.31,	8.19,	8.37,	3.35,	8.41,	7.52]

x = np.arange(len(labels))  # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
frame = ax.bar(2 * x - 0.5, frame_ccc, width, label='Frame')
face = ax.bar(2 * x, segment_ccc, width, label='Segment')
face2 = ax.bar(2 * x + 0.5, face2_ccc, width, label='Video')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('CCC (%)', fontsize=15)
# ax.set_title('Impression CCC scores by frame and face images')
ax.set_xticks([0, 2, 4, 6, 8, 10])
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


autolabel_label(frame, (-2, 1))
autolabel_label(face, (4, 1))
autolabel_label(face2, (2, 1))


fig.tight_layout()
plt.ylim(0, 70)
plt.savefig("Impression_frame_seg_vid_CCC.png", dpi=300)
plt.show()
