import matplotlib
import matplotlib.pyplot as plt
import numpy as np




class CompareGraph:
    def __init__(self, labels, input_1, input_2):
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars

        self.fig, self.ax = plt.subplots()
        self.frame = self.ax.bar(x - width / 2, input_1, width, label='Full frame')
        self.face = self.ax.bar(x + width / 2, input_2, width, label='Face frame')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        self.ax.set_ylabel('ACC (%)', fontsize=15)
        # ax.set_title('Impression ACC scores by frame and face images')
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels)
        self.ax.legend()

    def autolabel_frame(self, rects, xytxt=(-2, 2)):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            self.ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=xytxt,  # 3 points vertical offset
                        textcoords="offset points",
                        fontsize=8,
                        ha='center', va='bottom')


    def autolabel_face(self, rects, xytxt=(2, 2)):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            self.ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=xytxt,  # 3 points vertical offset
                        textcoords="offset points",
                        fontsize=8,
                        ha='center', va='bottom')

    def draw(self):
        self.autolabel_frame(self.frame)
        self.autolabel_face(self.face)

        self.fig.tight_layout()
        plt.ylim(85, 93)
        plt.savefig("Impression_ACC.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    labels = ['Ope', 'Con', 'Ext', 'Agr', "Neu", "Ave"]
    frame_acc = [91.01, 90.51, 90.50, 89.07, 90.46, 86.09, 89.18, 90.63, 89.68]
    face_acc = [90.75, 90.75, 91.13, 89.09, 89.48, 86.50, 90.03, 91.18, 89.86]
    graph = CompareGraph(labels, frame_acc, face_acc)
    graph.draw()
