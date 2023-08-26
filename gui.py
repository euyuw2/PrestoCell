
# file_browser_ui.py

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pickle
import sys
import os
import main
import numpy as np
import napari.utils.notifications as notif


class LineEdit(QWidget):

    def __init__(self, title, default=None):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        if default is not None:
            self.x = default
        else:
            self.x = '100'

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(self.x))
        self.lineEdit.textChanged.connect(self.text_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)

    def text_changed(self, s):
        self.x = s

    def getValue(self):
        return int(self.x)

# -------------------------------------------------------------------

class NucleiChannel(QWidget):


    def __init__(self, title, config, idx):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Label
        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(['0', '1'])
        if config is not None:
            self.comboBox.setCurrentIndex(int(config[idx]))
        # self.comboBox.currentTextChanged.connect(self.indexChanged)

        layout.addWidget(self.comboBox)

        # layout.addStretch()

    def getIndex(self):
        return self.comboBox.currentIndex()

# -------------------------------------------------------------------


class RunningMode(QWidget):

    def __init__(self, title, itemList, config, idx, pred_dir):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.pred_dir = pred_dir
        # Label
        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(itemList)
        self.warningLabel = QLabel()
        self.warningLabel.setStyleSheet("color:red;")

        if config is not None:
            self.comboBox.setCurrentIndex(int(config[idx]))
            if int(config[idx]) == 0:
                if os.path.exists(self.pred_dir):
                    self.warningLabel.setText('Existing predictions will be removed!')
                else:
                    self.warningLabel.setText('')
            else:
                if not os.path.exists(self.pred_dir):
                    self.warningLabel.setText('No existing predictions!')
                else:
                    self.warningLabel.setText('')
        self.comboBox.currentIndexChanged.connect(self.changeWarningText)
        layout.addWidget(self.warningLabel)
        layout.addWidget(self.comboBox)
        # layout.addWidget(self.warningPanel)

        # layout.addStretch()

    def getIndex(self):
        return self.comboBox.currentIndex()

    def changeWarningText(self, i):
        print(i, self.pred_dir)
        if i == 0:
            if os.path.exists(self.pred_dir):
                self.warningLabel.setText('Existing predictions will be removed!')
            else:
                self.warningLabel.setText('')
        else:
            if not os.path.exists(self.pred_dir):
                self.warningLabel.setText('No existing predictions!')
            else:
                self.warningLabel.setText('')


# -------------------------------------------------------------------

# class OtsuLaunch(QWidget):
#
#     def __init__(self, title, val):
#         QWidget.__init__(self)
#         layout = QHBoxLayout()
#         self.setLayout(layout)
#
#         self.otsu_coef = val
#
#         # Label
#         self.label = QLabel()
#         self.label.setText(title)
#         self.label.setFont(QFont("Arial", weight=QFont.Bold))
#         self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#         layout.addWidget(self.label)
#
#         self.lineEdit = QLineEdit()
#         self.lineEdit.setText(str(self.otsu_coef))
#         self.lineEdit.textChanged.connect(self.text_changed)
#
#         layout.addWidget(self.lineEdit)
#         # layout.addStretch()
#
#         self.hint = QLabel()
#         self.hint.setText("(0.0 - 10.0)")
#         self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
#
#         layout.addWidget(self.hint)
#
#     def text_changed(self, s):
#         self.otsu_coef = float(s)
#
#     def get_Otsu(self):
#         return self.otsu_coef

# -------------------------------------------------------------------

class PersiLaunch(QWidget):

    def __init__(self, title, config, idx):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        if config is not None:
            self.persi = config[idx]
        else:
            self.persi = '0.5'

        # Label
        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit()
        self.lineEdit.setText(str(self.persi))
        self.lineEdit.textChanged.connect(self.text_changed)

        layout.addWidget(self.lineEdit)
        # layout.addStretch()

        self.hint = QLabel()
        self.hint.setText("(0.0 - 1.0, separated by ';')")
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.hint)

    def text_changed(self, s):
        self.persi = s

    def getValue(self):
        return self.persi

# -------------------------------------------------------------------

class PixelSizeLaunch(QWidget):

    def __init__(self, config, idx):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        if config is not None:
            vals = config[idx].split(' ')
            self.x = vals[0]
            self.z = vals[1]
        else:
            self.x = '1.0'
            self.z = '1.0'

        # Label
        self.label_x = QLabel()
        self.label_x.setText('Pixel size (x, y)')
        self.label_x.setFont(QFont("Arial", weight=QFont.Bold))
        self.label_x.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit_x = QLineEdit()
        self.lineEdit_x.setText(self.x)
        self.lineEdit_x.textChanged.connect(self.text_changed_x)

        # layout.addStretch()

        self.label_z = QLabel()
        self.label_z.setText('Voxel depth (z)')
        self.label_z.setFont(QFont("Arial", weight=QFont.Bold))
        self.label_z.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit_z = QLineEdit()
        self.lineEdit_z.setText(self.z)
        self.lineEdit_z.textChanged.connect(self.text_changed_z)

        layout.addWidget(self.label_x)
        layout.addWidget(self.lineEdit_x)
        layout.addWidget(self.label_z)
        layout.addWidget(self.lineEdit_z)

    def text_changed_x(self, s):
        self.x = s


    def text_changed_z(self, s):
        self.z = s

    def getValue(self):
        return float(self.x), float(self.z)

# -------------------------------------------------------------------

class OtsuLaunch(QWidget):

    def __init__(self, config, idx):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        if config is not None:
            vals = config[idx].split(' ')
            self.x = vals[0]
            self.z = vals[1]
        else:
            self.x = '1.0'
            self.z = '1.0'

        # Label
        self.label_x = QLabel()
        self.label_x.setText('Otsu. cell')
        self.label_x.setFont(QFont("Arial", weight=QFont.Bold))
        self.label_x.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit_x = QLineEdit()
        self.lineEdit_x.setText(self.x)
        self.lineEdit_x.textChanged.connect(self.text_changed_x)

        # layout.addStretch()

        self.label_z = QLabel()
        self.label_z.setText('Otsu. nuclei')
        self.label_z.setFont(QFont("Arial", weight=QFont.Bold))
        self.label_z.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit_z = QLineEdit()
        self.lineEdit_z.setText(self.z)
        self.lineEdit_z.textChanged.connect(self.text_changed_z)

        layout.addWidget(self.label_x)
        layout.addWidget(self.lineEdit_x)
        layout.addWidget(self.label_z)
        layout.addWidget(self.lineEdit_z)


    def text_changed_x(self, s):
        self.x = s

    def text_changed_z(self, s):
        self.z = s

    def getValue(self):
        return float(self.x), float(self.z)

# -------------------------------------------------------------------

class PaddingLaunch(QWidget):

    def __init__(self, title, default=None):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        if default is not None:
            self.x = default
        else:
            self.x = '100'

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.lineEdit = QLineEdit()
        self.lineEdit.setText(self.x)
        self.lineEdit.textChanged.connect(self.text_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)

    def text_changed(self, s):
        self.x = s

    def getValue(self):
        return int(self.x)


# -------------------------------------------------------------------

class FileSize(QWidget):

    def __init__(self, s):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText('Input size (z, c, x, y)')
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        # self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.labelSize = QLabel()
        self.labelSize.setText(str(s))
        # self.labelSize.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.labelSize)

# -------------------------------------------------------------------

class FileSelector(QDialog):
    def __init__(self, default_path, save_dir, config, idx, parent=None):
        QDialog.__init__(self, parent)
        self.save_dir = save_dir
        self.fname = default_path.split('/')[-1]
        self.pred_dir = os.path.join(save_dir, self.fname)
        # Ensure our window stays in front and give it a title
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Presto Cell")
        # self.setFixedSize(500, 200)
        self.setFixedWidth(500)
        self.fname = None

        # Create and assign the main (vertical) layout.
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)

        layout = QHBoxLayout()
        self.setLayout(layout)
        self.browser_mode = 0
        self.filter_name = 'All files (*.*)'
        self.dirpath = QDir.currentPath()

        self.label = QLabel()
        self.label.setText('File Path')
        # self.label.setFixedWidth(65)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)
        self.lineEdit = QLineEdit(self)
        self.lineEdit.setFixedWidth(300)
        # self.lineEdit.setFixedHeight(100)
        self.lineEdit.setText(default_path)
        self.filepath = default_path
        self.lineEdit.textChanged.connect(self.text_changed)

        layout.addWidget(self.lineEdit)

        self.button = QPushButton('Browse')
        self.button.clicked.connect(self.getFile)
        layout.addWidget(self.button)

        vlayout.addLayout(layout)

        layout = QHBoxLayout()
        # Label
        self.label = QLabel()
        self.label.setText('Mode')
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.label)

        self.comboBox = QComboBox()
        self.comboBox.addItems(['Segmentation', 'Editing'])
        self.warningLabel = QLabel()
        self.warningLabel.setStyleSheet("color:red;")

        if config is not None:
            self.comboBox.setCurrentIndex(int(config[idx]))
            if int(config[idx]) == 0:
                if os.path.exists(self.pred_dir):
                    self.warningLabel.setText('Existing predictions will be removed!')
                else:
                    self.warningLabel.setText('')
            else:
                if not os.path.exists(self.pred_dir):
                    self.warningLabel.setText('No existing predictions!')
                else:
                    self.warningLabel.setText('')

        self.comboBox.currentIndexChanged.connect(self.changeWarningText)
        layout.addWidget(self.warningLabel)
        layout.addWidget(self.comboBox)
        vlayout.addLayout(layout)

        self.addButtonPanel(vlayout)

        self.show()

    def text_changed(self, s):
        self.filepath = s
        self.fname = s.split('/')[-1]
        self.pred_dir = os.path.join(self.save_dir, self.fname)
        if self.getModeIndex() == 0:
            if os.path.exists(self.pred_dir):
                self.warningLabel.setText('Existing predictions will be removed!')
            else:
                self.warningLabel.setText('')
        else:
            if not os.path.exists(self.pred_dir):
                self.warningLabel.setText('No existing predictions!')
            else:
                self.warningLabel.setText('')


    # --------------------------------------------------------------------
    def getFile(self):
        self.filepath = QFileDialog.getOpenFileName(self, caption='Choose File',
                                                          directory=self.dirpath,
                                                          filter=self.filter_name)[0]
        if not self.filepath:
            return
        else:
            self.lineEdit.setText(self.filepath)
            # --------------------------------------------------------------------

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)
        # --------------------------------------------------------------------

    def setlineEditWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    # --------------------------------------------------------------------
    def getPaths(self):
        return self.filepath

    def getModeIndex(self):
        return self.comboBox.currentIndex()

    def changeWarningText(self, i):
        print(i, self.pred_dir)
        if i == 0:
            if os.path.exists(self.pred_dir):
                self.warningLabel.setText('Existing predictions will be removed!')
            else:
                self.warningLabel.setText('')
        else:
            if not os.path.exists(self.pred_dir):
                self.warningLabel.setText('No existing predictions!')
            else:
                self.warningLabel.setText('')

    # --------------------------------------------------------------------
    def addButtonPanel(self, parentLayout):
        hlayout = QHBoxLayout()
        hlayout.addStretch()

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.buttonAction)
        hlayout.addWidget(self.button)
        parentLayout.addLayout(hlayout)

    # --------------------------------------------------------------------
    def buttonAction(self):
        print(self.getPaths())
        self.close()

    # --------------------------------------------------------------------


class Launch(QDialog):
    def __init__(self, shape, parent=None, config=None, idx=None):
        QDialog.__init__(self, parent)

        # Ensure our window stays in front and give it a title
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Presto Cell")
        # self.setFixedSize(400, 300)

        # Create and assign the main (vertical) layout.
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)

        self.inputSize = FileSize(shape)
        vlayout.addWidget(self.inputSize)

        self.nucleiCH = NucleiChannel('Nuclei channel', config, idx)
        vlayout.addWidget(self.nucleiCH)
        idx += 1

        self.otsu = OtsuLaunch(config, idx)
        vlayout.addWidget(self.otsu)
        idx += 1

        self.perciCell = PersiLaunch('Persi. cell', config, idx)
        vlayout.addWidget(self.perciCell)
        idx += 1

        # self.lowestPersiMode = RunningMode('Use Lowest Persi for Clustering', ['No', 'Yes'], config, 4)
        # vlayout.addWidget(self.lowestPersiMode)

        self.pixelSize = PixelSizeLaunch(config, idx)
        vlayout.addWidget(self.pixelSize)
        idx += 1

        if config is not None:

            self.paddingSize = PaddingLaunch('Extra Margin for Visualization (in pixels)', config[idx])
            vlayout.addWidget(self.paddingSize)
            idx += 1

            self.zBatch = PaddingLaunch('Num. of planes for nuclei detection', config[idx])
            vlayout.addWidget(self.zBatch)
            idx += 1

            self.diameter = PaddingLaunch('Nuclei diameter (in pixels)', config[idx])
            vlayout.addWidget(self.diameter)
            idx += 1

        else:
            self.paddingSize = PaddingLaunch('Visualization padding')
            vlayout.addWidget(self.paddingSize)

            self.zBatch = PaddingLaunch('z batch')
            vlayout.addWidget(self.zBatch)

            self.diameter = PaddingLaunch('Nuclei diameter')
            vlayout.addWidget(self.diameter)


        # self.runningMode = RunningMode('Mode', ['Segmentation', 'Editing'], config, idx)
        # vlayout.addWidget(self.runningMode)

        self.addButtonPanel(vlayout)

        self.show()

    # --------------------------------------------------------------------
    def addButtonPanel(self, parentLayout):
        hlayout = QHBoxLayout()
        hlayout.addStretch()

        self.button = QPushButton("OK")
        self.button.clicked.connect(self.buttonAction)
        hlayout.addWidget(self.button)
        parentLayout.addLayout(hlayout)

    # --------------------------------------------------------------------
    def buttonAction(self):
        # print(self.fileFB.getPaths())
        # print(self.nucleiCH.getIndex())
        # return self.fileFB.getPaths(), self.nucleiCH.getIndex()
        self.close()

    # --------------------------------------------------------------------


class EditingPanel(QWidget):
    def __init__(self, viewer):
        QWidget.__init__(self)
        self.viewer = viewer
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        self.editMode = 1
        self.addModeButton = QPushButton('ADD MODE')
        self.deleteModeButton = QPushButton('DELETE MODE')
        self.addModeButton.clicked.connect(self.button_add_mode)
        self.deleteModeButton.clicked.connect(self.button_delete_mode)
        vlayout.addWidget(self.addModeButton)
        vlayout.addWidget(self.deleteModeButton)

        self.saveAllButton = QPushButton('SAVE ALL')
        self.saveAllButton.clicked.connect(self.button_save_all)
        vlayout.addWidget(self.saveAllButton)

    def button_add_mode(self):
        self.editMode = 1

    def button_delete_mode(self):
        self.editMode = 0

    def button_save_all(self):
        pass

    def get_mode(self):
        return self.editMode


class OtsuPanelCell(QWidget):
    def __init__(self, otsu_coef, num_points, viewer, otsu_min=0, otsu_max=5):
        QWidget.__init__(self)
        self.viewer = viewer
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        vlayout.addWidget(StatsPanel('Num. of points.', num_points))
        self.sliderPanel = SliderPanel('Otsu coef.', otsu_min, otsu_max, otsu_coef)
        vlayout.addWidget(self.sliderPanel)

        self.button = QPushButton('RERUN')
        self.button.clicked.connect(self.button_click_run)
        vlayout.addWidget(self.button)

        self.button_accept = QPushButton('ACCEPT')
        self.button_accept.clicked.connect(self.button_click_accept)
        vlayout.addWidget(self.button_accept)

        self.accept = False

    def button_click_run(self):
        self.viewer.close()

    def button_click_accept(self):
        self.accept = True
        self.viewer.close()


class OtsuPanelNuclei(QWidget):
    def __init__(self, otsu_coef, num_points, viewer, otsu_min=0, otsu_max=5):
        QWidget.__init__(self)
        self.viewer = viewer
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        vlayout.addWidget(StatsPanel('Num. of points.', num_points))
        self.sliderPanel = SliderPanel('Otsu coef.', otsu_min, otsu_max, otsu_coef)
        vlayout.addWidget(self.sliderPanel)

        self.button = QPushButton('RERUN')
        self.button.clicked.connect(self.button_click_run)
        vlayout.addWidget(self.button)

        self.button_back = QPushButton('BACK')
        self.button_back.clicked.connect(self.button_click_back)
        vlayout.addWidget(self.button_back)

        self.button_accept = QPushButton('ACCEPT')
        self.button_accept.clicked.connect(self.button_click_accept)
        vlayout.addWidget(self.button_accept)

        self.accept = False
        self.back = False
    def button_click_run(self):
        self.viewer.close()

    def button_click_back(self):
        self.back = True
        self.viewer.close()

    def button_click_accept(self):

        self.accept = True
        self.viewer.close()

class StatsPanel(QWidget):
    def __init__(self, title, content):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.title = QLabel()
        self.title.setText(title)
        self.content = QLabel()
        self.content.setText(content)
        layout.addWidget(self.title)
        layout.addWidget(self.content)

class SliderPanel(QWidget):
    otsu = None

    def __init__(self, title, min_val, max_val, default_val, scale=10):
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.scale = scale
        self.otsu = default_val
        self.title = QLabel()
        self.title.setText(title)
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(min_val)
        self.sl.setMaximum(max_val * scale)
        self.sl.setValue(int(default_val * scale ))
        self.sl.valueChanged.connect(self.value_change)
        self.sl_val = QLabel()
        self.sl_val.setText(str(self.otsu))
        layout.addWidget(self.title)
        layout.addWidget(self.sl)
        layout.addWidget(self.sl_val)

    def value_change(self, i):
        self.otsu = i / self.scale
        self.sl_val.setText(str(self.otsu))


    # --------------------------------------------------------------------


class NucleiMaskPanel(QWidget):
    def __init__(self, v, viewer):
        QWidget.__init__(self)
        self.viewer = viewer
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        self.lineEdit = LineEdit('Diameter', v)
        vlayout.addWidget(self.lineEdit)

        self.button = QPushButton('RUN')
        self.button.clicked.connect(self.button_click_run)
        vlayout.addWidget(self.button)

        self.button_back = QPushButton('BACK')
        self.button_back.clicked.connect(self.button_click_back)
        vlayout.addWidget(self.button_back)

        self.button_accept = QPushButton('ACCEPT')
        self.button_accept.clicked.connect(self.button_click_accept)
        vlayout.addWidget(self.button_accept)

        self.state = -1
        self.back = False

    def button_click_run(self):
        self.state = 0
        self.viewer.close()

    def button_click_back(self):
        self.state = 2
        self.viewer.close()

    def button_click_accept(self):
        self.state = 1
        self.viewer.close()


class CellEditingPanel(QWidget):
    def __init__(self,):
        QWidget.__init__(self)
        self.setFixedHeight(200)
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        commandList = ['right arrow: next cell',
                       'left arrow: prev. cell',
                       'double-click: add/remove parts',
                       'right-click: re-cluster parts',
                       's: save edit',
                       'u: undo',
                       'a: show all cells',
                       'esc: close editing window'
                       ]
        for c in commandList:
            label = QLabel()
            label.setText(c)
            vlayout.addWidget(label)

        # self.button = QPushButton('Run')
        # self.button.clicked.connect(self.button_click_run)
        # vlayout.addWidget(self.button)
        #
        # self.button_accept = QPushButton('Accept')
        # self.button_accept.clicked.connect(self.button_click_accept)
        # vlayout.addWidget(self.button_accept)

        # self.state = -1


class CellEditingGoTo(QWidget):
    def __init__(self, idx, total, viewer, Visualizer):
        QWidget.__init__(self)
        self.viewer = viewer
        self.idx = int(idx) + 1
        self.jump = False
        self.Visualizer = Visualizer

        vlayout = QVBoxLayout()
        hlayout = QHBoxLayout()
        self.setLayout(vlayout)
        self.label = QLabel()
        self.label.setText('Cell index: ')
        # hlayout.addWidget(StatsPanel('Cell index', num_points))

        self.lineEdit = QLineEdit()
        self.lineEdit.setText(idx)
        self.lineEdit.textChanged.connect(self.text_change)
        self.labelTotal = QLabel()
        self.labelTotal.setText(f'/{total}')
        hlayout.addWidget(self.label)
        hlayout.addWidget(self.lineEdit)
        hlayout.addWidget(self.labelTotal)
        vlayout.addLayout(hlayout)
        self.button = QPushButton('GO TO')
        self.button.clicked.connect(self.button_click_run)
        vlayout.addWidget(self.button)

        self.accept = False

    def text_change(self, s):
        if s == '':
            self.idx = None
        else:
            self.idx = int(s)

    def button_click_run(self):
        self.Visualizer.cell_idx = self.idx
        self.Visualizer.reload_cell()

        # main.cell_idx = self.idx
        # self.jump = True
        # self.history = {}
        # f_cell_name = f'cell_{self.cell_idx}.pkl'
        # file_name = os.path.join(self.f_pred_dir, f_cell_name)
        # f_pred_clusters = os.path.join(self.f_interactive_dir, f_cell_name + '_pred_clusters.tiff')
        # self.viewer.title = f'{self.f_pred_dir}_cell_{self.cell_idx}/{self.num_cells}'
        # with open(file_name, 'rb') as f:
        #     dic_cell = pickle.load(f)
        #     pred_cropped = dic_cell['pred']
        #     self.pred_pbc = dic_cell['pbc']
        #     pred_cropped = [pred_cropped[0]]
        #     self.idx_mask_cell = dic_cell['idx']
        #     x_min, x_max, y_min, y_max, z_min, z_max = dic_cell['idx_boundary']
        #
        # self.x_cropped = self.x[z_min:z_max, :, x_min:x_max, y_min:y_max]
        # self.layer_data_list[0].data = self.x_cropped[:, 0, :, :]
        # self.layer_data_list[0].data = np.random.randint(0, 100, (10, 500, 500))
        # self.Visualizer.layer_data_list[0].data = np.random.randint(0, 100, (10, 500, 500))
        # self.Visualizer.

class CellEditingSave(QWidget):
    def __init__(self, Visualizer):
        QWidget.__init__(self)
        vlayout = QVBoxLayout()
        self.visualizer = Visualizer
        self.setLayout(vlayout)

        self.button_save_one = QPushButton('SAVE IN ONE CHANNEL')
        self.button_save_one.clicked.connect(self.button_save_one_click)
        vlayout.addWidget(self.button_save_one)

        self.button_save_separate = QPushButton('SAVE IN SEPARATE CHANNELS')
        self.button_save_separate.clicked.connect(self.button_save_separate_click)
        vlayout.addWidget(self.button_save_separate)

        self.accept = False
        self.back = False

    def button_save_one_click(self):
        self.visualizer.save_as_one()
        notif.show_info(f'Saved')

    def button_save_separate_click(self):
        self.visualizer.save_separate()
        notif.show_info(f'Saved')


class CellposeInstruction(QWidget):
    def __init__(self,):
        QWidget.__init__(self)
        self.setFixedHeight(200)
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)
        commandList = ['Make sure to go through\nall batches before accepting!',
                       # 'left arrow: prev. cell',
                       # 'double-click: add/remove parts',
                       # 'right-click: re-cluster parts',
                       # 's: save edit',
                       # 'u: undo',
                       # 'a: show all cells',
                       # 'esc: close editing window'
                       ]
        for c in commandList:
            label = QLabel()
            label.setText(c)
            vlayout.addWidget(label)


class ButtonExit(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        vlayout = QVBoxLayout()
        self.setLayout(vlayout)

        self.button_exit = QPushButton('EXIT')
        self.button_exit.clicked.connect(self.button_exit_click)
        vlayout.addWidget(self.button_exit)

    def button_exit_click(self):
        sys.exit()
# # ========================================================
# if __name__ == '__main__':
#     # Create the Qt Application

#     app = QApplication(sys.argv)
#     menu = Launch()
#     menu.show()
#     app.exec_()
#     fname = menu.fileFB.getPaths()
#     nuclei_ch = menu.nucleiCH.getIndex()
#     otsu_cell, otsu_nuclei = menu.otsu.getValue()
#     pixel_x, pixel_z = menu.pixelSize.getValue()
#     persi_str = menu.perciCell.getValue()
#     vis_pad_size = menu.paddingSize.getValue()
#     persi_cell_list = list(map(float, persi_str.split(';')))

