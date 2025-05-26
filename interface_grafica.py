import os
import sys
import cv2
import torch
import tkinter as tk
from tkinter import filedialog, Label, Button, Entry, StringVar, Text, messagebox, Frame
from PIL import Image, ImageTk
from threading import Thread

# ================== YOLO SETUP ==================
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov9'))
from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import LoadImages
from yolov9.utils.general import (
    non_max_suppression, scale_boxes, check_img_size
)
from yolov9.utils.torch_utils import select_device
from yolov9.utils.plots import colors

# ================== Configurações ==================
caminho_pesos = os.path.join('pesos', 'best.pt')
img_size = (416, 416)
device = select_device('')
model = DetectMultiBackend(caminho_pesos, device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = check_img_size(img_size, s=stride)


# ================== Funções ==================
def desenhar_caixas(im, det):
    for *xyxy, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        xyxy = [int(x) for x in xyxy]
        color = colors(int(cls), True)
        cv2.rectangle(im, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = xyxy[0] + t_size[0], xyxy[1] - t_size[1] - 3
        cv2.rectangle(im, (xyxy[0], xyxy[1]), c2, color, -1, cv2.LINE_AA)
        cv2.putText(im, label, (xyxy[0], xyxy[1] - 2),
                    0, 1, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    return im

def limitar_tamanho_input_video(*args):
    valor = threshold_video_var.get()
    if not valor.isdigit():
        threshold_video_var.set(''.join(filter(str.isdigit, valor)))
    elif len(valor) > 2:
        threshold_video_var.set(valor[:2])


# ================== Interface Tkinter ==================
janela = tk.Tk()
janela.title('Detector de Placas - Grupo 9')
janela.state('zoomed')
janela.configure(bg="#f0f0f0")

# ================== Telas ==================
frame_inicial = Frame(janela, bg="#f0f0f0")
frame_fotos = Frame(janela, bg="#f0f0f0")
frame_video = Frame(janela, bg="#f0f0f0")

for frame in (frame_inicial, frame_fotos, frame_video):
    frame.place(relwidth=1, relheight=1)


def mostrar_frame(frame):
    frame.tkraise()


# ================== Tela Inicial ==================
Label(frame_inicial, text="Detector de Placas - Grupo 9", font=("Arial", 28), bg="#f0f0f0").pack(pady=50)

Button(
    frame_inicial, text="Detectar Imagens", command=lambda: mostrar_frame(frame_fotos),
    font=("Arial", 16), width=25, height=2, bg="#4CAF50", fg="white"
).pack(pady=20)

Button(
    frame_inicial, text="Detectar Vídeos", command=lambda: mostrar_frame(frame_video),
    font=("Arial", 16), width=25, height=2, bg="#2196F3", fg="white"
).pack(pady=20)


# =====================================================================================
# ================== Tela de Detecção em Fotos ==================
# =====================================================================================
lista_imagens = []
resultados = []
imagens_originais = []
indice_atual = 0
threshold_conf = 0.25
modo_detectado = False


def detectar_placa(imagem_path, threshold_conf):
    dataset = LoadImages(imagem_path, img_size=img_size, stride=stride, auto=pt)
    texto_placa = ""
    imagem_saida = None

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, threshold_conf, 0.45, None, False, max_det=1000)

        for det in pred:
            im0 = im0s.copy()
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                chars = []
                for *xyxy, conf, cls in det:
                    x1 = int(xyxy[0])
                    label = str(names[int(cls)]) if int(cls) < len(names) else str(int(cls))
                    chars.append((x1, label))

                im0 = desenhar_caixas(im0, det)
                chars = sorted(chars, key=lambda x: x[0])
                texto_placa = ''.join([str(c[1]) for c in chars])

            imagem_saida = im0

    return imagem_saida, texto_placa



def selecionar_imagens():
    global lista_imagens, resultados, indice_atual, imagens_originais, modo_detectado
    lista_imagens = filedialog.askopenfilenames(filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")])
    resultados = []
    imagens_originais = []
    indice_atual = 0
    modo_detectado = False
    if lista_imagens:
        for caminho in lista_imagens:
            imagem_cv = cv2.imread(caminho)
            if imagem_cv is not None:
                imagens_originais.append(imagem_cv)
        botao_detectar.config(state=tk.NORMAL)
        exibir_imagem()


def iniciar_detectar():
    global resultados, threshold_conf, indice_atual, modo_detectado
    resultados = []
    indice_atual = 0
    modo_detectado = True

    try:
        valor = int(threshold_var.get())
        threshold_conf = max(0, min(valor / 100, 1))
    except ValueError:
        messagebox.showerror("Erro", "Insira um número válido para o threshold.")
        return

    label_status.config(text="🔍 Detectando...")
    janela.update()

    for imagem in lista_imagens:
        resultado = detectar_placa(imagem, threshold_conf)
        resultados.append(resultado)

    label_status.config(text="")
    exibir_imagem()


def exibir_imagem():
    global indice_atual
    if modo_detectado and resultados:
        imagem, placa = resultados[indice_atual]
    elif imagens_originais:
        imagem = imagens_originais[indice_atual].copy()
        placa = ''
    else:
        return

    if imagem is not None:
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_pil = Image.fromarray(imagem_rgb).resize((800, 600))
        imagem_tk = ImageTk.PhotoImage(imagem_pil)

        label_imagem.config(image=imagem_tk)
        label_imagem.image = imagem_tk

        label_resultado.config(text=f'{placa} ({indice_atual + 1}/{len(imagens_originais)})')
        label_navegacao.config(
            text="← Anterior     Próxima →" if len(imagens_originais) > 1 else ""
        )


def proxima_imagem(event=None):
    global indice_atual
    if imagens_originais:
        indice_atual = (indice_atual + 1) % len(imagens_originais)
        exibir_imagem()


def imagem_anterior(event=None):
    global indice_atual
    if imagens_originais:
        indice_atual = (indice_atual - 1) % len(imagens_originais)
        exibir_imagem()


def limitar_tamanho_input(*args):
    valor = threshold_var.get()
    if not valor.isdigit():
        threshold_var.set(''.join(filter(str.isdigit, valor)))
    elif len(valor) > 2:
        threshold_var.set(valor[:2])


# Layout fotos
frame_topbar = tk.Frame(frame_fotos, bg="#f0f0f0")
frame_topbar.pack(pady=15, fill="x")

Button(frame_topbar, text="← Voltar", command=lambda: mostrar_frame(frame_inicial),
       font=('Arial', 12), bg="#9E9E9E", fg="white", width=10).grid(row=0, column=0, padx=(10, 20))

botao_selecionar = Button(frame_topbar, text='Selecionar Imagens', command=selecionar_imagens,
                           font=('Arial', 14), bg="#4CAF50", fg="white", width=18)
botao_selecionar.grid(row=0, column=1, padx=20)

Label(frame_topbar, text="Threshold (%):", font=('Arial', 14), bg="#f0f0f0").grid(row=0, column=2, padx=5)

threshold_var = StringVar()
threshold_var.trace('w', limitar_tamanho_input)

Entry(frame_topbar, textvariable=threshold_var, width=5, font=('Arial', 14), justify='center').grid(row=0, column=3)

botao_detectar = Button(frame_topbar, text='Detectar', command=iniciar_detectar,
                         font=('Arial', 14), state=tk.DISABLED,
                         bg="#2196F3", fg="white", width=12)
botao_detectar.grid(row=0, column=4, padx=20)

label_status = Label(frame_fotos, text='', font=('Arial', 14), fg="blue", bg="#f0f0f0")
label_status.pack(pady=5)

label_imagem = Label(frame_fotos, bg="white", bd=2, relief="solid")
label_imagem.pack(pady=10)

label_resultado = Label(frame_fotos, text='', font=('Arial', 16), bg="#f0f0f0")
label_resultado.pack(pady=2)  # Reduzi o espaçamento aqui

label_navegacao = Label(frame_fotos, text='', font=('Arial', 14), bg="#f0f0f0")
label_navegacao.pack(pady=(0, 5))  # Adicionei um espaçamento para subir mais

janela.bind('<Right>', proxima_imagem)
janela.bind('<Left>', imagem_anterior)


# =====================================================================================
# ================== Tela de Detecção em Vídeo ==================
# =====================================================================================
video_rodando = False
video_selecionado = None


def resetar_video():
    global video_rodando, video_selecionado
    video_rodando = False
    video_selecionado = None
    label_video.config(image='')
    label_video.image = None
    log_video.config(state=tk.NORMAL)
    log_video.delete(1.0, tk.END)
    log_video.config(state=tk.DISABLED)
    botao_parar.config(state=tk.DISABLED)
    botao_detectar_video.config(state=tk.DISABLED)


def selecionar_video():
    global video_selecionado
    caminho = filedialog.askopenfilename(filetypes=[("Vídeo", "*.mp4 *.avi *.mov *.mkv")])
    if caminho:
        video_selecionado = caminho
        cap = cv2.VideoCapture(caminho)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (600, 400))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.config(image=imgtk)
            label_video.image = imgtk
        cap.release()
        log_video.config(state=tk.NORMAL)
        log_video.delete(1.0, tk.END)
        log_video.config(state=tk.DISABLED)
        botao_detectar_video.config(state=tk.NORMAL)


def processar_video_na_tela():
    global video_rodando
    if not video_selecionado:
        messagebox.showerror("Erro", "Selecione um vídeo primeiro.")
        return

    try:
        valor = int(threshold_video_var.get())
        threshold_video = max(0, min(valor / 100, 1))
    except ValueError:
        messagebox.showerror("Erro", "Insira um número válido para o threshold.")
        return

    cap = cv2.VideoCapture(video_selecionado)
    video_rodando = True
    botao_parar.config(state=tk.NORMAL)

    frame_count = 0
    frame_skip = 4

    while cap.isOpened() and video_rodando:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            frame_resized = cv2.resize(frame, (600, 400))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.config(image=imgtk)
            label_video.image = imgtk
        else:
            im = cv2.resize(frame, img_size)
            im = im[:, :, ::-1].transpose(2, 0, 1).copy()
            im = torch.from_numpy(im).to(device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = model(im, augment=False, visualize=False)[0]
            pred = non_max_suppression(pred, threshold_video, 0.45, None, False, max_det=1000)

            texto_placa = ''
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    frame = desenhar_caixas(frame, det)

                    chars = []
                    for *xyxy, conf, cls in det:
                        x1 = int(xyxy[0])
                        label = str(names[int(cls)]) if int(cls) < len(names) else str(int(cls))
                        chars.append((x1, label))

                    chars = sorted(chars, key=lambda x: x[0])
                    texto_placa = ''.join([str(c[1]) for c in chars])

            if texto_placa:
                log_video.config(state=tk.NORMAL)
                log_video.insert(tk.END, f"Frame {frame_count}: Placa detectada {texto_placa}\n")
                log_video.see(tk.END)
                log_video.config(state=tk.DISABLED)

            frame_resized = cv2.resize(frame, (600, 400))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label_video.config(image=imgtk)
            label_video.image = imgtk

        janela.update()

    cap.release()
    resetar_video()


def iniciar_video_na_tela():
    if not video_rodando:
        Thread(target=processar_video_na_tela).start()


def parar_video():
    global video_rodando
    video_rodando = False


# Layout vídeo
Label(frame_video, text="Detecção em Vídeo na Tela", font=("Arial", 24), bg="#f0f0f0").pack(pady=20)

frame_botoes_video = tk.Frame(frame_video, bg="#f0f0f0")
frame_botoes_video.pack(pady=10)

Button(frame_botoes_video, text="← Voltar", command=lambda: [resetar_video(), mostrar_frame(frame_inicial)],
       font=('Arial', 14), width=10, height=2, bg="#9E9E9E", fg="white").grid(row=0, column=0, padx=10)

Button(frame_botoes_video, text="Selecionar Vídeo", command=selecionar_video,
       font=('Arial', 14), width=18, height=2, bg="#4CAF50", fg="white").grid(row=0, column=1, padx=10)

botao_detectar_video = Button(frame_botoes_video, text="Detectar", command=iniciar_video_na_tela,
                               font=('Arial', 14), width=12, height=2, bg="#2196F3", fg="white", state=tk.DISABLED)
botao_detectar_video.grid(row=0, column=2, padx=10)

botao_parar = Button(frame_botoes_video, text="Parar", command=parar_video,
                      font=('Arial', 14), width=10, height=2, bg="#F44336", fg="white", state=tk.DISABLED)
botao_parar.grid(row=0, column=3, padx=10)

Label(frame_botoes_video, text="Threshold (%):", font=('Arial', 14), bg="#f0f0f0").grid(row=0, column=4, padx=5)

threshold_video_var = StringVar()
threshold_video_var.trace('w', limitar_tamanho_input_video)
Entry(frame_botoes_video, textvariable=threshold_video_var, width=5, font=('Arial', 14), justify='center').grid(row=0, column=5, padx=5)

label_video = Label(frame_video, bg="white", bd=2, relief="solid", width=600, height=400)
label_video.pack(pady=10)

Label(frame_video, text="Log de Detecções", font=("Arial", 14), bg="#f0f0f0").pack()

log_video = Text(frame_video, height=8, width=100, state=tk.DISABLED)
log_video.pack(pady=5)

# ================== Inicializa ==================
mostrar_frame(frame_inicial)
janela.mainloop()