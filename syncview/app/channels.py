"""Editable channel table: one row per plotted trace."""
import json

from PySide6 import QtCore, QtWidgets

from ..core.filters import MODES, params

COLS = ["Show", "Label", "Ch", "Ref", "Mode", "Low Hz", "High Hz", "Notch", "Extra", "Y range"]
C_SHOW, C_LABEL, C_CH, C_REF, C_MODE, C_LO, C_HI, C_NOTCH, C_EXTRA, C_Y = range(len(COLS))
EXTRA_KEYS = ["order", "smooth_ms", "env_lp", "plot_fs", "win_s"]


def _num(s):
    s = s.strip()
    if s.lower() in ("", "none", "-"):
        return None
    return float(s)


def fmt_ylim(ylim):
    return "auto" if not ylim else f"{ylim[0]:.4g}, {ylim[1]:.4g}"


class ChannelPanel(QtWidgets.QWidget):
    """Emits specs_changed(list_of_specs) after any edit (debounced)."""
    specs_changed = QtCore.Signal(list)

    def __init__(self, ch_names, parent=None):
        super().__init__(parent)
        self.ch_names = ch_names
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.AllEditTriggers)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        for c, w in zip(range(len(COLS)), (38, 110, 64, 64, 86, 56, 60, 50, 110, 90)):
            self.table.setColumnWidth(c, w)
        hdr.setStretchLastSection(True)
        self.table.itemChanged.connect(self._edited)
        self.table.setToolTip(
            "Edit cells directly.  Low/High Hz = filter band (blank = none).  Notch: e.g. 60 or 60,180.\n"
            "Extra: key=value for order, smooth_ms, env_lp, plot_fs, win_s.  Y range: 'auto' or 'lo, hi'.\n"
            "Ctrl+wheel (⌘+scroll on macOS) over a trace scales its Y range; double-click a trace resets it to auto.")

        btns = QtWidgets.QHBoxLayout()
        for text, fn in [("Add", self.add_row), ("Remove", self.remove_row), ("▲", lambda: self.move(-1)),
                         ("▼", lambda: self.move(1)), ("Load…", self.load_preset), ("Save…", self.save_preset)]:
            b = QtWidgets.QToolButton(text=text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.table)
        lay.addLayout(btns)

        self._debounce = QtCore.QTimer(self, singleShot=True, interval=400)
        self._debounce.timeout.connect(lambda: self.specs_changed.emit(self.specs()))
        self._loading = False

    def sizeHint(self):
        return QtCore.QSize(460, 500)

    # ------------------------------------------------------------------ rows <-> specs
    def set_specs(self, specs):
        self._loading = True
        self.table.setRowCount(0)
        for s in specs:
            self._append(s)
        self._loading = False
        self.specs_changed.emit(self.specs())

    def _append(self, s):
        r = self.table.rowCount()
        self.table.insertRow(r)
        p = params(s)
        show = QtWidgets.QTableWidgetItem()
        show.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        show.setCheckState(QtCore.Qt.Checked if s.get("show", True) else QtCore.Qt.Unchecked)
        self.table.setItem(r, C_SHOW, show)
        self.table.setItem(r, C_LABEL, QtWidgets.QTableWidgetItem(s.get("label", s["ch"])))
        self.table.setCellWidget(r, C_CH, self._combo(self.ch_names, s["ch"]))
        self.table.setCellWidget(r, C_REF, self._combo(["—"] + self.ch_names, s.get("ref") or "—"))
        self.table.setCellWidget(r, C_MODE, self._combo(MODES, s["mode"]))
        lo, hi = p["band"]
        self.table.setItem(r, C_LO, QtWidgets.QTableWidgetItem("" if lo is None else f"{lo:g}"))
        self.table.setItem(r, C_HI, QtWidgets.QTableWidgetItem("" if hi is None else f"{hi:g}"))
        notch = p.get("notch")
        notch = ",".join(f"{float(v):g}" for v in (notch if isinstance(notch, (list, tuple)) else [notch])) \
            if notch else ""
        self.table.setItem(r, C_NOTCH, QtWidgets.QTableWidgetItem(notch))
        extra = " ".join(f"{k}={s[k]:g}" for k in EXTRA_KEYS if k in s and s[k] is not None)
        self.table.setItem(r, C_EXTRA, QtWidgets.QTableWidgetItem(extra))
        self.table.setItem(r, C_Y, QtWidgets.QTableWidgetItem(fmt_ylim(s.get("ylim"))))

    def _combo(self, items, current):
        cb = QtWidgets.QComboBox()
        cb.addItems(items)
        cb.setCurrentText(current)
        cb.currentTextChanged.connect(self._edited)
        return cb

    def _row_spec(self, r):
        t = lambda c: (self.table.item(r, c).text() if self.table.item(r, c) else "")
        s = dict(ch=self.table.cellWidget(r, C_CH).currentText(),
                 mode=self.table.cellWidget(r, C_MODE).currentText(),
                 label=t(C_LABEL),
                 show=self.table.item(r, C_SHOW).checkState() == QtCore.Qt.Checked)
        ref = self.table.cellWidget(r, C_REF).currentText()
        if ref != "—":
            s["ref"] = ref
        try:
            s["band"] = (_num(t(C_LO)), _num(t(C_HI)))
        except ValueError:
            pass
        try:
            notch = [float(v) for v in t(C_NOTCH).replace(";", ",").split(",") if v.strip()]
            if notch:
                s["notch"] = notch
        except ValueError:
            pass
        for kv in t(C_EXTRA).replace(",", " ").split():
            k, _, v = kv.partition("=")
            if k in EXTRA_KEYS:
                try:
                    s[k] = int(v) if k == "order" else float(v)
                except ValueError:
                    pass
        y = t(C_Y).replace(";", ",").split(",")
        try:
            if len(y) == 2:
                s["ylim"] = (float(y[0]), float(y[1]))
        except ValueError:
            pass
        return s

    def specs(self):
        return [self._row_spec(r) for r in range(self.table.rowCount())]

    def set_ylim(self, row, ylim):
        """Update the Y-range cell without triggering a rebuild."""
        self._loading = True
        self.table.item(row, C_Y).setText(fmt_ylim(ylim))
        self._loading = False

    # ------------------------------------------------------------------ editing
    def _edited(self, *_):
        if not self._loading:
            self._debounce.start()

    def add_row(self):
        r = self.table.currentRow()
        s = self._row_spec(r) if r >= 0 else dict(ch=self.ch_names[0], mode="slow")
        s.pop("ylim", None)
        self._append(s)
        self._edited()

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._edited()

    def move(self, d):
        r = self.table.currentRow()
        specs = self.specs()
        if 0 <= r < len(specs) and 0 <= r + d < len(specs):
            specs[r], specs[r + d] = specs[r + d], specs[r]
            self.set_specs(specs)
            self.table.selectRow(r + d)

    def load_preset(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load channel preset", "", "JSON (*.json)")
        if fn:
            self.window().apply_preset(json.loads(open(fn).read()))

    def save_preset(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save channel preset", "", "JSON (*.json)")
        if fn:
            if not fn.endswith(".json"):
                fn += ".json"
            with open(fn, "w") as f:
                json.dump(self.window().preset(), f, indent=1)
