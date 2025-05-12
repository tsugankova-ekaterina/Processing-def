
# -*- coding: cp1251 -*-

import sys, os
import pandas as pd
import csv
import matlab.engine
from datetime import datetime,  timedelta
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QApplication
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from main import Ui_MainWindow  # ������ ��������� ����
from apply_comp import Ui_Dialog as Ui_CompCh_Dialog
from gaussian import Ui_Dialog as Ui_Gaussian_Dialog
from mean import Ui_Dialog as Ui_Mean_Dialog
from emd import Ui_Dialog as Ui_EMD_Dialog
from hht import Ui_Dialog as Ui_HHT_Dialog
from view_trace import Ui_Dialog as Ui_ViewTrace_Dialog
from open_file_dialog import Ui_Dialog as Ui_FileOpenDialog 
from load_earthq_file_dialog import Ui_Dialog as Ui_FileEQLoadDialog
from nn import Ui_Dialog as Ui_NN_Dialog
from save_channel import Ui_Dialog as Ui_Save_Dialog
from load_csv import Ui_Dialog as Ui_Load_CSV_Dialog

from matplotlib.figure import Figure
from scipy.signal import hilbert
from matplotlib import cm  # ��� �������� ����

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

class LoadCSVDialog(QtWidgets.QDialog, Ui_Load_CSV_Dialog):
    def __init__(self, signals, current_signals, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.signals = signals
        self.current_signals = current_signals
        self.toolButton.clicked.connect(self.select_file)
        self.buttonBox.accepted.connect(self.load_csv)

    def select_file(self):
        """��������� ������ ������ ����� � ��������� ���� � lineEdit."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select the CSV file", "", "CSV Files (*.csv)"
        )
        if file_path:
            self.lineEdit.setText(file_path)

    def load_csv(self):
        """��������� ������ �� CSV-����� � ��������� �� � signals."""
        file_path = self.lineEdit.text()
        channel_name = self.lineEdit_2.text()

        if not file_path:
            QtWidgets.QMessageBox.warning(self, "Error", "Specify the file name.")
            return

        if not channel_name:
            QtWidgets.QMessageBox.warning(self, "Error", "Specify the channel name.")
            return

        # �������� ������������ ����� ������
        if channel_name in self.signals:
            QtWidgets.QMessageBox.warning(self, "Error", "The channel name already exists")
            return

        try:
            # ������ ������ �� CSV-�����
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # ���������� ���������
                data = [float(row[1]) for row in reader]  # ������������, ��� ������ �� ������ �������

            # ��������� ������ � signals
            self.signals[channel_name] = {"data": np.array(data)}
            self.current_signals.extend([channel_name])
            QtWidgets.QMessageBox.information(self, "Success", "Data uploaded successfully.")
            self.accept()  # ��������� ������
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to upload file: {str(e)}")

class SaveChannelDialog(QtWidgets.QDialog, Ui_Save_Dialog):
    def __init__(self, signals, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.signals = signals
        self.comboBox.addItems(self.signals.keys())
        # self.populate_combo_box()
        self.buttonBox.accepted.connect(self.save_to_csv)

    def save_to_csv(self):
        """��������� ��������� ������ � CSV-����."""
        selected_channel = self.comboBox.currentText()
        file_name = self.lineEdit.text()

        if not file_name:
            QtWidgets.QMessageBox.warning(self, "Error", "Specify the file name.")
            return

        # �������� ������ ��� ���������� ������
        data = self.signals[selected_channel]['data']

        # ��������� ������ ������ ����������
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a directory")

        if not directory:
            return  # ������������ ������� ����� ����������

        # ��������� ������ � CSV-����
        file_path = f"{directory}/{file_name}.csv"
        try:
            with open(file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Index", "Value"])  # ���������
                for index, value in enumerate(data):
                    writer.writerow([index, value])
            QtWidgets.QMessageBox.information(self, "Success", f"File saved: {file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Couldn't save the file: {str(e)}")

class NN_Dialog(QtWidgets.QDialog, Ui_NN_Dialog):
    def __init__(self, signals, start_index, end_index):
        super().__init__()
        self.setupUi(self)
        self.signals = signals
        self.start_index = start_index
        self.end_index  = end_index

        # ��������� ���������� ���������
        self.fill_comboboxes()

        # ���������� ������
        self.pushButtonClose.clicked.connect(self.close)
        self.pushButtonApply.clicked.connect(self.apply_neural_networks)

        # �������� ������
        try:
            # �������� ������� ����������
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.network_1 = load_model(os.path.join(current_dir, '..', 'networks', 'best_model_sc_ga.keras'))
            self.network_2 = load_model(os.path.join(current_dir, '..', 'networks','bm_imf8_9.keras'))
            self.network_3 = load_model(os.path.join(current_dir, '..', 'networks','autoencoder_sig910aug_mse.keras'))
        except Exception as e:
            print(f"������ ��� �������� ������: {e}")

    def fill_comboboxes(self):
        """��������� ���������� ���������� ��������."""
        for combo in [self.comboBox, self.comboBox_2, self.comboBox_3]:
            combo.clear()
            combo.addItem("None")  # ��������� ����� "None"
            combo.addItems(self.signals)  # ��������� �������

    def apply_neural_networks(self):
        """��������� ��������� � ��������� ��������."""
        # �������� ��������� ������
        channel_1 = self.comboBox.currentText()
        channel_2 = self.comboBox_2.currentText()
        channel_3 = self.comboBox_3.currentText()

        self.label_NN_1.setText(str(''))
        self.label_NN_2.setText(str(''))
        self.label_NN_3.setText(str(''))

        # �������� ���������� ����� ���������� ����
        if self.end_index - self.start_index != 21612:  
            QtWidgets.QMessageBox.warning(self.dialog, "Error", "The number of points in the time series should be 21,612")
            return

        # ���������, ��� ������� ������
        if channel_1 != "None":
            result = self.apply_nn_1(self.signals[channel_1]['data'][self.start_index:self.end_index])
            self.label_NN_1.setText(str(result))

        if  channel_2 != "None":
            result = self.apply_nn_2(self.signals[channel_2]['data'][self.start_index:self.end_index])
            self.label_NN_2.setText(str(result))

        if  channel_3 != "None":
            result = self.apply_nn_3(self.signals[channel_3]['data'][self.start_index:self.end_index])
            self.label_NN_3.setText(str(result))

    def apply_nn_1(self, data):
        # �������� � ��������� MinMaxScaler 
        scaler = MinMaxScaler(feature_range=(0, 1))

        # ��������������� ������
        scaled_data = scaler.fit_transform(data.reshape(1, -1))

        # ������������ �� �������� �������
        y_pred_prob = self.network_1.predict(scaled_data)
        y_pred = (y_pred_prob >= 0.5).astype(int)

        return y_pred[0][0]

    def apply_nn_2(self, data):
        # ������������ �� �������� �������
        y_pred_prob = self.network_2.predict(data.reshape(1,-1))#(1, -1, 1)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        return y_pred[0][0]

    def apply_nn_3(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))

        # ��������������� ������
        scaled_data = scaler.fit_transform(data.reshape(1, -1))
        x_predict = self.network_3.predict(scaled_data)
        mse = np.mean(np.power(scaled_data - x_predict, 2), axis=1)
        
        MSE_threshold = 0.012650916928523813
        if mse > MSE_threshold:
            return 0
        else:
            return 1

class HHTDialog(QtWidgets.QDialog, Ui_HHT_Dialog):
    def __init__(self, signals, fs, start_date, end_date, start_index, end_index, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.signals = signals
        self.fs = fs
        self.start_date = start_date
        self.start_index = start_index
        self.end_date = end_date
        self.end_index  = end_index

        # ���������� ���������� ���������� ���������
        self.comboBox.addItems(self.signals.keys())

        # ��������� �������
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.graphicsView.setLayout(QtWidgets.QVBoxLayout())
        self.graphicsView.layout().addWidget(self.canvas)

        # ����������� ������
        self.pushButton.clicked.connect(self.close)
        self.pushButton_2.clicked.connect(self.plot_spectrum)

    def closeEvent(self, event):
        # ������� ������ �� ��� ���� �� ������ � ������������ ������
        if self.parent() and hasattr(self.parent(), "spectrum_dialogs"):
            self.parent().spectrum_dialogs.remove(self)
        event.accept()

    def update_info(self):
        """��������� ���������� � label."""
        self.info.setText(f"Spectrum ({self.comboBox.currentText()})<p>{self.start_date} - {self.end_date}<\p>")

    def plot_spectrum(self):
        """������ ������ ��� ���������� ������."""
        try:
            
            # �������� ��������� �����
            channel = self.comboBox.currentText()
            data = self.signals[channel]['data'][self.start_index:self.end_index]
            self.update_info()
            
            # ��������� �������������� ��������� 
            analytic_signal = hilbert(data)
            amplitude = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi) * self.fs

            # �����
            t = np.arange(len(data)) / self.fs

            # �������� ������� �� ����������� �������
            t = t[:-1]  # ������� ��������� �������, ����� ������ ������ � instantaneous_frequency
            amplitude = amplitude[:-1] 

            # �������� �������� �� ���������� (� �������)
            left_edge = self.spinBox.value()  # ����� ������� (������)
            right_edge = self.spinBox_2.value()  # ������ ������� (������)

            # ����������� ������� � ������� (��)
            frequency_limits = [1.0 / (60 * right_edge), 1.0 / (60 * left_edge)]

            # ��������� ������ �� �������
            mask = (instantaneous_frequency >= frequency_limits[0]) & (instantaneous_frequency <= frequency_limits[1])
            instantaneous_frequency_filtered = instantaneous_frequency[mask]
            amplitude_filtered = amplitude[mask]
            t_filtered = t[mask]

            # ������� ������
            self.figure.clear()

            # ������ ������
            
            ax = self.figure.add_subplot(111)

            # ������ ������ ������� �� ������� � �������� ���������� ���������
            scatter = ax.scatter(t_filtered, instantaneous_frequency_filtered, c=amplitude_filtered, cmap=cm.viridis, s=10)

            # ��������� �������� �����
            self.figure.colorbar(scatter, ax=ax, label="Amplitude")

            ax.set_ylim(frequency_limits[0], frequency_limits[1])

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            
            self.figure.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05)
            # ��������� ����� ������������ ������ �������
            self.figure.tight_layout()
            
            # ��������� ������� ����� �� ����
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            
            # ��������� ������
            self.canvas.draw()
            
        except Exception as e:
            # ��������� ������
            print(e)
            QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

class EMDDialog(QtWidgets.QDialog,Ui_EMD_Dialog):
    # ������ ��� �������� ������
    data_ready = QtCore.pyqtSignal(dict)

    def __init__(self, signals):
        super().__init__()
        self.setupUi(self) 
        self.signals = signals
        self.engine = matlab.engine.start_matlab()  # ������ MATLAB Engine
        self.imfs = None  # ��� �������� ����������� ������������
        # ������ ��� �������� ���� ��������� �������
        self.existing_channel_names = []
        self.new_channel_name = None
        self.selected_imf = None
        self.selected_channel = None
        self.dec_red = False
        
        # ������ ����������� � ������ "OK" �����������
        self.radioButtonAll.setEnabled(False)
        self.radioButtonNew.setEnabled(False)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        
        # ��������� QComboBox ���������� �������
        self.comboBoxCH.addItems([key for key in self.signals.keys()])

        # ���������� ������ "Decompose"
        self.pushButtonDecompose.clicked.connect(self.decompose)

        # �������������� ��������� ������ "OK"
        self.ok_button = self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok)
        self.ok_button.clicked.disconnect()  # ��������� ����������� ���������
        self.ok_button.clicked.connect(self.on_accept)  # ���������� ��� �����

    def decompose(self):
        """������������ ������� ������ 'Decompose'."""
        # �������� ������ �� �����
        sift_tolerance = self.doubleSpinBox.value()
        sift_max_iter = self.spinBoxMaxIt.value()
        max_num_extrema = self.spinBoxNumExtr.value()
        max_num_imf = self.spinBoxNumIMF.value()
        self.selected_channel = self.comboBoxCH.currentText()

        # �������� ������ ��� ���������� ������
        data = self.signals[self.selected_channel]['data']#[self.start_index:self.end_index]
        data = np.array(data, dtype=np.float64)

        # �������� ������� EMD ����� MATLAB Engine
        try:
            self.imfs = self.engine.emd(
                matlab.double(data.tolist()),
                'SiftRelativeTolerance', sift_tolerance,
                'SiftMaxIterations', sift_max_iter,
                'MaxNumExtrema', max_num_extrema,
                'MaxNumIMF', max_num_imf
            )
            self.imfs = np.array(self.imfs)
            self.dec_red = True

            # ���������� ����������� � ������ "OK"
            self.radioButtonAll.setEnabled(True)
            self.radioButtonNew.setEnabled(True)
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)

            # ��������� ��������� IMF
            self.comboBoxIMF.clear()

            for i in range(self.imfs.shape[1]):
                self.comboBoxIMF.addItem(f"IMF {i + 1}")
            # self.comboBoxIMF.addItem("residual")

            QtWidgets.QMessageBox.information(self, "Success", "Decomposition completed successfully!")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Decomposition failed: {str(e)}")
    
    def on_accept(self):
        """������������ ������� ������ 'OK'."""
        if not self.dec_red:
            QtWidgets.QMessageBox.warning(self, "Warning", "No decomposition results available. Please perform decomposition first.")
            return
        
        # �������� ������ ��� ��������
        results = self.get_results()

        # ����������, ����� ����������� �������
        if self.radioButtonAll.isChecked():
            # �������� ������ ��� ���� �������

            print(f"Processing")
            QtWidgets.QMessageBox.information(self, "Info", "Data has been sent for all IMFs.")
            # ��������� ���� � ���������� ������
            self.data_ready.emit(results)  
            self.accept()  
            
        elif self.radioButtonNew.isChecked():
            # �������� ������ ��� ������ ������
            self.new_channel_name = self.lineEdit.text()  # ��� ������ ������ �� lineEdit
            if not self.new_channel_name:
                QtWidgets.QMessageBox.warning(self, "Warning", "Please enter a name for the new channel.")
                return
            
            # �������� �� ������������ �����
            if self.new_channel_name in self.existing_channel_names or self.new_channel_name in self.signals.keys():
                self.new_channel_name = self.generate_unique_name(self.new_channel_name)
                QtWidgets.QMessageBox.information(self, "Info", f"The name was already taken. Using '{self.new_channel_name}' instead.")
                self.lineEdit.setText(self.new_channel_name)

            # ��������� ����� ��� ������ � ������ ������������
            self.existing_channel_names.append(self.new_channel_name)

            results['selected_imf'] = self.comboBoxIMF.currentIndex()
            results['new_channel_name'] = self.new_channel_name

            print(f"Creating new channel")
            self.data_ready.emit(results) 
            QtWidgets.QMessageBox.information(self, "Info", "New channel data has been sent. You can continue working.")
            return

    def generate_unique_name(self, base_name):
        """���������� ���������� ���, �������� �������, ���� ��� ��� ����������."""
        counter = 1
        new_name = base_name
        while new_name in self.existing_channel_names or new_name in self.signals.keys():
            new_name = f"{base_name}_{counter}"
            counter += 1
        return new_name

    def get_results(self):
        """���������� ���������� ������������ � ��������� ���������."""
        return {
            'imfs': self.imfs,
            'selected_imf': self.selected_imf, #comboBoxIMF.currentText(),
            'selected_channel': self.selected_channel, #comboBoxCH.currentText(),
            'new_channel_name' : self.new_channel_name,
            'radio_all' : self.radioButtonAll.isChecked(),
            'radio_new' : self.radioButtonNew.isChecked()
        }

    def closeEvent(self, event):
        """��������� MATLAB Engine ��� �������� ����."""
        self.engine.quit()
        event.accept()

class ViewTraceDialog(QtWidgets.QDialog, Ui_ViewTrace_Dialog):
    def __init__(self, channels, current_ch):
        super().__init__()
        self.setupUi(self) 
        self.channels = channels
        self.current_ch = current_ch
        self.selected_channels = []

        # ��������� �������
        self.tableWidget.setColumnCount(2)  # ��� �������: ����� � �������
        self.tableWidget.setHorizontalHeaderLabels(["Channel", "Choose"])
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        self.checkBoxSelAll.stateChanged.connect(self.toggle_select_all)

        # ���������� ������� �������� � ����������
        self.tableWidget.setRowCount(len(channels))
        for i, channel in enumerate(channels):
            # ��� ������
            item = QtWidgets.QTableWidgetItem(channel)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)  # ��������� ��������������
            self.tableWidget.setItem(i, 0, item)

            # �������
            checkbox = QtWidgets.QCheckBox(self)
            # checkbox.setChecked(True)
            checkbox.setChecked(channel in current_ch)
            checkbox.stateChanged.connect(self.update_selection)
            self.tableWidget.setCellWidget(i, 1, checkbox)

        # ����������� ������
        # self.buttonBox.accepted.connect(self.accept)
        
        self.buttonBox.accepted.connect(self.accept_and_update)
        self.buttonBox.rejected.connect(self.reject)

        # ����������� �������� "Unload unused traces"
        self.checkBoxUnload.stateChanged.connect(self.update_unload_decision)

        # ���������� ��� �������� ������� �� �������� �������������� �����
        self.unload_unused_traces = False

        # ��������� ��������� �������� "Select All" ��� �������������
        self.update_select_all_state()

    def accept_and_update(self):
        """������������ ������� ������ 'OK' � ��������� ��������� ������."""
        self.update_selection()  # ��������� ��������� ������ ����� ���������
        self.accept()  # ��������� ������


    def toggle_select_all(self):
        """������������ ��������� �������� 'Select All'."""
        if self.checkBoxSelAll.isChecked():
            # ������������� ��� �������� � ��������� "������"
            for i in range(0, self.tableWidget.rowCount()):
                checkbox = self.tableWidget.cellWidget(i, 1)
                checkbox.setChecked(True)
        else:
            # ������� ��������� �� ���� ���������
            for i in range(0, self.tableWidget.rowCount()):
                checkbox = self.tableWidget.cellWidget(i, 1)
                checkbox.setChecked(False)
        self.update_selection()

    def update_selection(self):
        """��������� ������ ��������� �������."""
        self.selected_channels = []
        for i in range(self.tableWidget.rowCount()):
            checkbox = self.tableWidget.cellWidget(i, 1)
            if checkbox.isChecked():
                self.selected_channels.append(self.tableWidget.item(i, 0).text())
        self.update_select_all_state()

    def update_select_all_state(self):
        """��������� ��������� �������� 'Select All'."""
        all_checked = all(self.tableWidget.cellWidget(i, 1).isChecked() for i in range(self.tableWidget.rowCount()))
        self.checkBoxSelAll.stateChanged.disconnect(self.toggle_select_all)
        self.checkBoxSelAll.setChecked(all_checked)
        self.checkBoxSelAll.stateChanged.connect(self.toggle_select_all)

    def update_unload_decision(self):
        """��������� ������� �� �������� �������������� �����."""
        self.unload_unused_traces = self.checkBoxUnload.isChecked()

    def get_selected_channels(self):
        """���������� ������ ��������� �������."""
        return self.selected_channels

    def get_unload_decision(self):
        """���������� ������� �� �������� �������������� �����."""
        return self.unload_unused_traces

class ApplyMeanDialog(QtWidgets.QDialog, Ui_Mean_Dialog):
    def __init__(self, channel_names):
        super().__init__()
        self.setupUi(self)  # ��������� ���������� �������

        # ��������� QComboBox ���������� �������
        self.comboBox.addItems(channel_names)

        # ����������� �������� � �������
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def get_data(self):
        apply_to_all = self.checkBox.isChecked()
        selected_channel = self.comboBox.currentText() if not apply_to_all else None
        return selected_channel, apply_to_all 

class ApplyGaussianDialog(QtWidgets.QDialog, Ui_Gaussian_Dialog):
    def __init__(self, channel_names):
        super().__init__()
        self.setupUi(self)  # ��������� ���������� �������

        # ��������� QComboBox ���������� �������
        self.comboBox.addItems(channel_names)

        # ����������� �������� � �������
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def get_data(self):
        # �������� ������ �� ��������� ����������
        selected_channel = self.comboBox.currentText()
        noise_level = self.doubleSpinBox.value()
        
        # ���������, ����� ����������� �������
        apply_to_all = self.radioButton.isChecked()
        save_as_new_trace = self.radioButton_2.isChecked()

        # ���� ������� ������ �����������, �������� �������� �� QLineEdit
        new_trace_name = self.lineEdit.text() if save_as_new_trace else None
        return {
            'selected_channel': selected_channel,
            'noise_level': noise_level,
            'apply_to_all': apply_to_all,
            'save_as_new_trace': save_as_new_trace,
            'new_trace_name': new_trace_name
        }

class ApplyCopmChDialog(QtWidgets.QDialog, Ui_CompCh_Dialog):
    def __init__(self, channel_names):
        super().__init__()
        self.setupUi(self)  # ��������� ���������� �������

        # ��������� QComboBox ���������� �������
        self.CH1.addItems(channel_names)
        self.CH2.addItems(channel_names)
        self.compCH.addItems(channel_names)

        # ����������� �������� � �������
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def get_data(self):
        K = self.doubleSpinBox.value()  # ���������� �� .value()
        ch1_comp_name = self.comp_name1.text()
        ch2_comp_name = self.comp_name2.text()
        ch1_selected = self.CH1.currentText()  # �������� ��������� ������� �� CH1
        ch2_selected = self.CH2.currentText()  # �������� ��������� ������� �� CH2
        comp_selected = self.compCH.currentText()  
        return K, ch1_selected, ch2_selected, comp_selected, ch1_comp_name, ch2_comp_name

class FileOpenDialog(QtWidgets.QDialog, Ui_FileOpenDialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # ��������� ���������� �������
        
        self.ch1 = None  # ������� ��� �������� ������
        self.ch2 = None  
        self.comp_ch = None  
        self.T = None  # ������ �������������
        self.begintime = None  # ���� ������ ������
        self.timesec = None # ����� � ������ ������ (�)
        
        # ����������� �������� � �������
        self.buttonBox.accepted.connect(self.validate_and_read_files)
        self.buttonBox.rejected.connect(self.reject)
    

    def get_data(self):
        ch1_path = self.lineEditCH1.text()
        ch2_path= self.lineEditCH2.text()
        comp_ch_path = self.lineEditCompCh.text()
        metadata = self.lineEditMeta.text()
        return ch1_path, ch2_path, comp_ch_path, metadata

    def validate_and_read_files(self):
        ch1_path, ch2_path, comp_ch_path, metadata = self.get_data()
        file_paths = [ch1_path, ch2_path, comp_ch_path, metadata]
        
        for path in file_paths:
            if not os.path.isfile(path):
                QtWidgets.QMessageBox.warning(self, "Error", f"File not exists: {path}")
                return 

        try:# ������ ����� � �����������
            # ������ ����������
            with open(metadata, 'r') as fid:
                data = np.loadtxt(fid, delimiter='\t', dtype=str)  # ������ ������ � �������
                self.begintime = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S')
                self.T = data[1].astype(float)  # ������ �������������
                self.timesec = data[2].astype(float)  # ����� � ������ ������
                self.metabt = data[3].astype(int)  # ���������� � ������
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read file: {metadata}\n{str(e)}")
            print(str(e))
            return
        
        self.ch1 = self.read_sig_file(ch1_path)
        self.ch2 = self.read_sig_file(ch2_path)
        self.comp_ch = self.read_sig_file(comp_ch_path)

        self.accept()  # ���� ��� ���� �����, ��������� ������
        # �������� ����� � ������
        
    def read_sig_file(self, path):
        try:
            # ������ ������ � �������
            with open(path, 'rb') as fid:
                # ������ ���������� (��������, 8 ����)
                np.fromfile(fid, dtype=np.int8, count=self.metabt)
                # ������ ������ 
                data = np.fromfile(fid, dtype=np.int16)  # ��������������, ��� ������ ����� ������ int16
                return data
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read file: {path}\n{str(e)}")
            print(str(e))
            return
             
class FileEQLoadDialog(QtWidgets.QDialog, Ui_FileEQLoadDialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # ��������� ���������� �������
        
        self.tab= None
        # ����������� ��������
        self.buttonBox.rejected.connect(self.reject)
        # self.buttonBox.accepted.setEnabled(False)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Open).setEnabled(False)
        self.pushButton.clicked.connect(self.load_preview_data)

    def load_preview_data(self):
        # ��������� ������ �� ����� �����
        EQ_path = self.lineEdit.text()

        if not os.path.isfile(EQ_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"File not exists: {EQ_path}")
            return
        try:               
            # ������ ������ �� ���������� �����
            self.tab = pd.read_csv(EQ_path, sep='\t', header=None, names=['Time', 'latitude', 'longitude', 'class'], 
                              parse_dates=['Time'], date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S'))
            self.populate_table(self.tab)
            self.buttonBox.button(QtWidgets.QDialogButtonBox.Open).setEnabled(True)  # Enable Open button
                
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to read file: {EQ_path}\n{str(e)}")
            print(str(e))
            return
    

    def populate_table(self, data):
        model = QStandardItemModel(data.shape[0], data.shape[1])
        model.setHorizontalHeaderLabels(data.columns)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = QStandardItem(str(data.iat[i, j]))
                model.setItem(i, j, item)

        self.tableView.setModel(model)

        # �������������� ���������� ������ ������� ��� ����������
        for column in range(model.columnCount()):
            self.tableView.resizeColumnToContents(column)


class PlotWidget(QWidget):
    def __init__(self, data, signal_key, start_date, end_date, start=None, end=None, eqdot=None):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # ������� ������
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        # ���������� ������
        self.plot_data(data, signal_key, start_date, end_date, start, end, eqdot)
        
        self.setFixedSize(1050, 250)
        

    def plot_data(self, data, signal_key, start_date, end_date, start=None, end=None, eqdot=None):
        if len(data) == 0:
            print("������: ������ ������ ������. ������ �� ����� ��������.")
            return
        
        ax = self.figure.add_subplot(111)
        ax.clear()

        # ��������� ����� �������
        self.figure.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.1)

        ax.plot(data)
        ax.set_ylim([np.min(data), np.max(data)])  # ������������� ������� �� ��� Y
        # ������� ��� X
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=8)

        # ��������� ������ ��� Y
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ��������� ���� � ������� ����
        ax.text(0.01, 0.95, start_date, transform=ax.transAxes, fontsize=8, verticalalignment='top')
        ax.text(0.99, 0.95, end_date, transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')

        # ��������� ��������, ������� � ������� �������� � ������ ������ ����
        max_val = np.max(data)
        min_val = np.min(data)
        avg_val = np.mean(data)
        stats_text = f"max:{max_val:.2f},min:{min_val:.2f},avg:{avg_val:.2f}"
        ax.text(0.99, 0.05, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='right')

        ax.text(0.01, 0.05, signal_key, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', horizontalalignment='left')

         # ������ �������������� ����� ������������� � ������������ ������, ���� ��������� ��������
        if start is not None and end is not None:
            ax.axvspan(start, end, color='gray', alpha=0.3)  # �������������� ����� �������������
        if eqdot is not None:
            ax.axvline(x=eqdot, color='gray', linestyle='--')  # ������������ ����� ������

        self.canvas.draw()

    def closeEvent(self, event):
        plt.close(self.figure)  # ��������� ������ ��� �������� �������
        event.accept()

# ������� ��� ��������� �������
def getdot(N, begin, time, T):
    dur = (time - begin).total_seconds()  # ������� �� ������� � ��������
    t = round(dur / T)  # ���������� ������ ������� �� ������ �������������
    # ���� ������ ��������� � �������� �������
    if 0 < t <= N:
        return t
    else:
        return -1

def edit_EQtab(EQtab, N, begin, T):
    # �������� ������ �������
    size = EQtab.shape[0]
    erthquakes = []

    # ��������� ������ erthquakes
    for i in range(size):
        dot = getdot(N, begin, EQtab['Time'].iloc[i], T)
        erthquakes.append(dot)

    erthquakes = np.array(erthquakes)

    # ������� ����� �������
    two_days_ago = np.round(erthquakes - 60 * 60 * 24 * 2 / T).astype(int)
    one_and_a_half_days_ago = np.round(erthquakes - 60 * 60 * 24 * 1.5 / T).astype(int)

    # ��������� ����� ������� � �������
    EQtab['eqdot'] = erthquakes
    EQtab['twodago'] = two_days_ago
    EQtab['onedago'] = one_and_a_half_days_ago

    # ������� �������, �� �������� ������� � ���������� ������
    toDelete = EQtab['twodago'] < 1
    EQtab = EQtab[~toDelete]
    return EQtab

def apply_gaussian_noise(signal, noise_level):
    """
    ����������� ����������� ��� �� ������.
    :param signal: �������� ������ (numpy.array).
    :param noise_level: ������� ���� (����������� ����������).
    :return: ������ � ���������� �����.
    """
    noise = np.random.normal(0, noise_level, signal.shape)
    return signal + noise

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.signals = {}  # ������� ��� �������� �������� � �� ����������
        self.current_signals = []  # ������ ��� ������������ ������� ��������, ������������ �� ��������
        self.channels_to_remove_mean = [] # ������ �������, ��� ������� ����� ������� �������
        self.spectrum_dialogs = []
  
        self.T = None  # ������ �������������
        self.fs = None
        self.begin_time = None # ���� ������ ������
        self.end_time = None # ���� ��������� ������
        # self.timesec = None # ����� � ������ ������
        self.EQtab = None # ������� � ���������
        self.N = None # ���������� ����� ����
        self.start_date_show = None
        self.start_index_show = None
        self.end_date_show = None
        self.end_index_show = None

        # ������� ����� ��� scrollAreaWidgetContents
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.scrollAreaWidgetContents.setLayout(self.scroll_layout)

        self.actionOpen.triggered.connect(self.open_file_dialog)
        self.actionEarthquakes.triggered.connect(self.load_file_dialog)
        self.actionApply_compCH.triggered.connect(self.apply_comp_ch)
        self.actionApply_Gaussian_noise.triggered.connect(self.apply_gaussian)
        self.actionRemove_mean.triggered.connect(self.remove_mean)
        self.actionTrace_Filter.triggered.connect(self.trace_filter_dialog)
        self.actionEMD.triggered.connect(self.show_emd_dialog)
        self.actionSpectrum.triggered.connect(self.open_spectrum_dialog)
        self.actionNeural_network.triggered.connect(self.open_nn_dialog)
        self.actionSave_as_csv.triggered.connect(self.show_save_dialog)
        self.actionFile_csv.triggered.connect(self.show_load_csv_dialog)

        # ���������� ����������� �������
        self.radioButtonEvent.toggled.connect(self.handle_update_plots)
        self.radioButtonDate.toggled.connect(self.handle_update_plots)
        self.checkBox.toggled.connect(self.handle_update_plots)
        self.dateTimeEditBegin.dateTimeChanged.connect(self.handle_update_plots)
        self.dateTimeEditEnd.dateTimeChanged.connect(self.handle_update_plots)
        self.comboBox.currentIndexChanged.connect(self.handle_update_plots)  # ���������� ��������� ������ � comboBox
    
    def show_load_csv_dialog(self):
        if self.T is None  or self.fs is None or self.begin_time is None or self.end_time is None:
            QtWidgets.QMessageBox.warning(self, "Error", "Unable to upload files.")
            return
        else:
            dialog = LoadCSVDialog(self.signals, self.current_signals, self)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                self.update_plots(count=1)

    def show_save_dialog(self):
        """��������� ���������� ���� ��� ���������� ������."""
        dialog = SaveChannelDialog(self.signals, self)
        dialog.exec_()

    def open_nn_dialog(self):
        if not self.signals:
            QtWidgets.QMessageBox.critical(self, "Error", f"No data available")
            return
        if self.start_index_show is None or self.end_index_show is None:
            num  = 21612
            QtWidgets.QMessageBox.critical(self, "Error", f"Output a time series of {num} points on a graph")
            return
        dialog = NN_Dialog(self.signals, self.start_index_show, self.end_index_show)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            print("NN end")
        
    def open_spectrum_dialog(self):
        """��������� ���������� ���� ��� ���������� �������."""
        dialog = HHTDialog(self.signals, self.fs, self.start_date_show, self.end_date_show, self.start_index_show, self.end_index_show)
        dialog.show() 
        self.spectrum_dialogs.append(dialog)
        # dialog.exec_()

    def show_emd_dialog(self):
        dialog = EMDDialog(self.signals)#, self.start_index_show, self.end_index_show)
        # ���������� ������ data_ready � �����
        dialog.data_ready.connect(self.handle_emd_data)
        
        dialog.exec_()

    def handle_emd_data(self, data):
        """������������ ������, ���������� �� �������."""
        
        if data['radio_all']:
            imf = data['imfs']
            ch_name = data['selected_channel']
            try:
                for i in range(imf.shape[1]):
                    self.signals[f"{ch_name}_IMF{i}"] = {'data': imf[:,i]}#, 'T': self.T, 'start_index': 0, 'end_index': self.N}
                    self.current_signals.extend([f"{ch_name}_IMF{i}"])
                self.update_plots(count=imf.shape[1])
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"You alredy decompose this channel (try another name): {str(e)}")


        elif data['radio_new']:
            index_imf = data['selected_imf']
            imf = data['imfs'][:,index_imf]
            new_ch = data['new_channel_name']
            self.signals[new_ch] = {'data': imf}#, 'T': self.T, 'start_index': 0, 'end_index': self.N}
            self.current_signals.extend([new_ch])
            self.update_plots(count=1)

        print("Received data from EMD dialog:", data)

    def trace_filter_dialog(self):
        channels = [key for key in self.signals.keys()]
        current_ch = self.current_signals
        dialog = ViewTraceDialog(channels, current_ch)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            selected_channels = dialog.get_selected_channels()
            unload_decision = dialog.get_unload_decision()
            if unload_decision:
                if not selected_channels:  # �������� �� ������� ��������� �������
                    QtWidgets.QMessageBox.warning(self, "Warning", "No channels selected for unloading.")
                    return
                for signal_key in list(self.signals.keys()):  # ���������� list(), ����� �������� ������ ��������� ������� �� ����� ��������
                    if signal_key not in selected_channels:
                        print(signal_key)
                        self.signals.pop(signal_key)

            self.current_signals = selected_channels
            self.update_plots()

    def remove_mean(self):
        channel_names = self.current_signals
        
        if channel_names is None or len(channel_names)==0:
            QtWidgets.QMessageBox.warning(self, "Error", f"The signals are not loaded")
            return
        dialog = ApplyMeanDialog(channel_names)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            ch, apply_to_all = dialog.get_data()
            
            if apply_to_all:
                for channel in channel_names:
                    if channel not in self.channels_to_remove_mean:
                # ��������� ��� ������ � ������
                        self.channels_to_remove_mean = channel_names
            else:
                # ��������� ������ ��������� �����
                if ch not in self.channels_to_remove_mean:
                    self.channels_to_remove_mean.append(ch)
            
            # ��������� �������
            self.update_plots()


    def apply_gaussian(self):
        channel_names = [key for key in self.signals.keys()]
        
        if channel_names is None or len(channel_names)==0:
            QtWidgets.QMessageBox.warning(self, "Error", f"The signals are not loaded")
            return
        dialog = ApplyGaussianDialog(channel_names)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            data = dialog.get_data()
            self.apply_gaussian_noise_to_signals(data)
        

    def apply_gaussian_noise_to_signals(self, data):
        selected_channel = data['selected_channel']
        noise_level = data['noise_level']
        apply_to_all = data['apply_to_all']
        save_as_new_trace = data['save_as_new_trace']
        new_trace_name = data['new_trace_name']

        if apply_to_all:
            # �������� ��� �� ��� �������
            for channel_name in self.signals:
                self.signals[channel_name]['data'] = apply_gaussian_noise(self.signals[channel_name]['data'], noise_level)
            self.update_plots()
        elif save_as_new_trace:
            # ������� ����� ������ � ���������� �����
            if new_trace_name not in self.signals:
                new_signal = self.signals[selected_channel].copy()
                new_signal['data'] = apply_gaussian_noise(new_signal['data'], noise_level)
                
                self.signals[new_trace_name] = new_signal
                self.current_signals.extend([new_trace_name])
                self.update_plots(count=1)
            else:
                QtWidgets.QMessageBox.warning(self, "Error", f"Signal name '{new_trace_name}' already exists or empty.")
                return
        else:
            # �������� ��� ������ �� ��������� ������
            self.signals[selected_channel]['data'] = apply_gaussian_noise(self.signals[selected_channel]['data'], noise_level)
            self.update_plots()

    def handle_update_plots(self):
        # �������� update_plots ��� ����������
        self.update_plots()


    def apply_comp_ch(self):
        # �������� �������� ����������� �������
        channel_names = [key for key in self.signals.keys()]
        
        if channel_names is None or len(channel_names)==0:
            QtWidgets.QMessageBox.warning(self, "Error", f"The signals are not loaded")
            return

        dialog = ApplyCopmChDialog(channel_names)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            K, ch1_selected, ch2_selected, comp_selected, ch1_comp_name, ch2_comp_name = dialog.get_data()
            new1 = self.signals[ch1_selected]['data']-K*self.signals[comp_selected]['data']
            new2 = self.signals[ch2_selected]['data']-K*self.signals[comp_selected]['data']
            if ch1_comp_name in self.signals:
                QtWidgets.QMessageBox.warning(self, "Error", f"Signal name '{ch1_comp_name}' already exists.")
                return

            if ch2_comp_name in self.signals:
                QtWidgets.QMessageBox.warning(self, "Error", f"Signal name '{ch2_comp_name}' already exists.")
                return

            # ��������� ����� ������� � self.signals
            self.signals[ch1_comp_name] = {
                'data': new1
            }

            self.signals[ch2_comp_name] = {
                'data': new2
            }

            count = 2
            # ��������� ������� ������� ��� �����������
            self.current_signals.extend([ch1_comp_name, ch2_comp_name])

            # ��������� �������
            self.update_plots(count)


    def update_plots(self, count=None):
        # ���������, ��� ������ ���������
        if not self.signals:
            return

        start_rect = None
        end_rect = None
        eqdot = None
        start_date = None
        end_date= None
        start_index= None
        end_index= None

        if self.radioButtonDate.isChecked():

            # �������� ��������� ����
            start_date = self.dateTimeEditBegin.dateTime().toPyDateTime()
            end_date = self.dateTimeEditEnd.dateTime().toPyDateTime()

            # ���������, ��� ���� ��������� � �������� begin_time � end_time
            if start_date < self.begin_time or end_date > self.end_time:
                QtWidgets.QMessageBox.warning(self, "Error", "Selected date range is out of bounds.")
                return

            # ����������� ���� � ������� �������
            start_index = int((start_date - self.begin_time).total_seconds() / self.T)
            end_index = int((end_date - self.begin_time).total_seconds() / self.T)+1

            # ���������, ��� ������� ���������
            if start_index >= end_index:
                QtWidgets.QMessageBox.warning(self, "Error", "Incorrect date range: the start date is later than the end date.")
                return

        elif self.radioButtonEvent.isChecked() and self.EQtab is not None:# and self.ch1 is not None:
            # ���������, ��� comboBox ����� ��������� �������
            if self.comboBox.currentIndex() == -1:
                QtWidgets.QMessageBox.warning(self, "Error", "No earthquake selected in the comboBox.")
                return

            # �������� ������ ���������� ��������
            selected_index = self.comboBox.currentIndex()
            end_index = self.EQtab.iloc[selected_index]['onedago']+1
            start_index = self.EQtab.iloc[selected_index]['twodago']
            
            if not self.checkBox.isChecked():
                # �������� �������� ��� �������������� � ������������ �����
                eqdot = self.EQtab.iloc[selected_index]['eqdot']                
                start_rect = start_index
                end_rect = end_index
                # �������� �������� �� ������� EQtab
                end_index = self.EQtab.iloc[selected_index]['eqdot']+int(6*3600/self.T)
                start_index = start_index-int(6*3600/self.T)

                eqdot = eqdot-start_index              
                start_rect = start_rect-start_index
                end_rect = end_rect-start_index                

            # ���������, ��� ������� ���������
            if end_index < 0 or start_index < 0 or end_index >= self.N or start_index >= self.N:
                QtWidgets.QMessageBox.warning(self, "Error", "Invalid index range in EQtab.")
                return

            # ��������� ������� � �������������� �������� end_index � start_index
            start_date = self.begin_time + timedelta(seconds=start_index * self.T)
            end_date = self.begin_time + timedelta(seconds=end_index * self.T)

        if start_date is not None:
            self.lenX.setText(f"({end_index-start_index})")
            self.start_date_show = start_date
            self.start_index_show = start_index
            self.end_date_show = end_date
            self.end_index_show = end_index

        # ������� ���������� �������
        if count is None:
            self.clear_plots()
            # ��������� ������� ������ ��� ������� ��������
            for signal_key in self.current_signals:
                if signal_key in self.signals:
                    # �������� �� ���� ��������
                    data = self.signals[signal_key]['data'].copy()
                    # ���� ����� � ������ ��� ��������� ��������, �������� ��� �� ��������� ���������
                    if signal_key in self.channels_to_remove_mean:
                        mean_value = np.mean(data[start_index:end_index])
                        data[start_index:end_index] = data[start_index:end_index] - mean_value
                    
                    # self.channels_to_remove_mean = {}
                    self.plot_signal(data[start_index:end_index], signal_key, start_date, end_date, start_rect, end_rect, eqdot)
        else:
            for signal_key in self.current_signals[-count:]:
                if signal_key in self.signals:

                    data = self.signals[signal_key]['data'].copy()
                    # ���� ����� � ������ ��� ��������� ��������, �������� ��� �� ��������� ���������
                    if signal_key in self.channels_to_remove_mean:
                        mean_value = np.mean(data[start_index:end_index])
                        data[start_index:end_index] = data[start_index:end_index] - mean_value
                    
                    # self.channels_to_remove_mean = {}
                    self.plot_signal(data[start_index:end_index], signal_key, start_date, end_date, start_rect, end_rect, eqdot)
        
    def clear_plots(self):
        # ������� ��� ������� �� scroll_layout
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                # ��������� ������ Matplotlib
                if hasattr(widget, 'figure'):
                    plt.close(widget.figure)
                widget.deleteLater()
        # ������������� ������������ �������
        QApplication.processEvents()  
            
    def open_file_dialog(self):
        dialog = FileOpenDialog()

        if dialog.exec_() == QtWidgets.QDialog.Accepted:

            ch1 = dialog.ch1 
            ch2 = dialog.ch2
            comp_ch = dialog.comp_ch
            self.T = dialog.T
            self.fs = 1/self.T; # ������� �������������
            timesec = dialog.timesec
            self.begin_time = dialog.begintime
            self.end_time = self.begin_time + timedelta(seconds=timesec.max())
            self.N = ch1.shape[0]
            
            # ��������� ����������� ������� � �������
            self.signals['ch1'] = {'data': ch1}
            self.signals['ch2'] = {'data': ch2}
            self.signals['comp_ch'] = {'data': comp_ch}

            # ��������� ������� �������
            self.current_signals = ['ch1', 'ch2', 'comp_ch']
            if self.EQtab is not None:
                self.EQtab = edit_EQtab(self.EQtab, self.N, self.begin_time, self.T)
                self.update_ui_with_EQ()
            
            self.update_ui_with_data()  # ��������� ��������� � ������ �������

    def update_ui_with_data(self):
        
        # ����������� ������ ��� �����������
        start_time_str = self.begin_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = self.end_time.strftime('%Y-%m-%d %H:%M:%S')

        # ��������� ����� info
        self.info.setText(f"������: {start_time_str}, ���������: {end_time_str}")
        
        # ��������� �������� � QDateTimeEdit
        self.dateTimeEditBegin.dateTimeChanged.disconnect(self.handle_update_plots)
        self.dateTimeEditEnd.dateTimeChanged.disconnect(self.handle_update_plots)
        self.radioButtonDate.toggled.disconnect(self.handle_update_plots)
        self.dateTimeEditBegin.setDateTime(QtCore.QDateTime(self.begin_time))
        self.dateTimeEditEnd.setDateTime(QtCore.QDateTime(self.end_time))
        self.radioButtonDate.toggled.connect(self.handle_update_plots)
        self.dateTimeEditBegin.dateTimeChanged.connect(self.handle_update_plots)
        self.dateTimeEditEnd.dateTimeChanged.connect(self.handle_update_plots)
        

    def plot_signal(self, data, signal_key, start_date, end_date, start=None, end=None, eqdot=None):#signal_key, start_date, end_date, start=None, end=None, eqdot=None):
        plot_widget = PlotWidget(data, signal_key, start_date, end_date, start, end, eqdot)
        self.scroll_layout.addWidget(plot_widget)
        
    def update_ui_with_EQ(self):
        # ���������, ���� �� ������ � �������
        if self.EQtab is not None and not self.EQtab.empty:
            self.comboBox.currentIndexChanged.disconnect(self.handle_update_plots)
            # ������� ��������� ����� �����������
            self.comboBox.clear()

            # �������� ������ �� ������� "Time" � ��������� �� � ���������
            time_values = self.EQtab['Time'].tolist()  # ����������� � ������
            self.comboBox.addItems([str(time) for time in time_values])  # ��������� ��������
            self.comboBox.currentIndexChanged.connect(self.handle_update_plots)

    def load_file_dialog(self):
        dialog = FileEQLoadDialog()
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.EQtab  = dialog.tab
            if self.N is not None and self.begin_time is not None and self.T is not None:
                self.EQtab = edit_EQtab(self.EQtab, self.N, self.begin_time, self.T)
            
            self.update_ui_with_EQ()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())