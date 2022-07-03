from awesometkinter.bidirender import add_bidi_support
import audio
import model
import tkinter as tk
import webrtcvad


RATE = 16000


if __name__ == '__main__':
    kerasmodel = model.load_model()

    root = tk.Tk()
    root.title('Keyword Spotting')

    frm = tk.Frame(root, padx=50, pady=20)
    frm.grid()

    lbl = tk.Label(frm, pady=25, text='{CLICK RECORD TO START}')
    lbl.grid(column=0, row=0)
    add_bidi_support(lbl)

    vad = webrtcvad.Vad(3)

    def record():
        lbl.set('{RECORDING}')
        lbl.update()
        signal = audio.record()
        audio.save(signal)
        lbl.set('{INFERING}')
        lbl.update()
        lbl.set(model.predict(kerasmodel, signal, RATE))

    tk.Button(frm, text='Record', command=record).grid(column=0, row=1)

    root.mainloop()
