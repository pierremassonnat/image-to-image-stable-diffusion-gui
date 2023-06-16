from compel import Compel
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from diffusers import DPMSolverMultistepScheduler
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QLineEdit,QFormLayout,QGridLayout,QLabel,QScrollArea,QHBoxLayout,QFileDialog,QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap,QPalette

class LogIm2im:
    def __init__(self,rn):
        self.init_system(rn)
        self.init_ihm()

    def init_system(self,rn):
        self.pipe = StableDiffusionImg2ImgPipeline.from_ckpt(rn, torch_dtype=torch.float16)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.safety_checker = None
        self.pipe.requires_safety_checker = False
        self.pipe.enable_attention_slicing()
        self.pipe = self.pipe.to("cuda")
        self.compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder)
        self.generator = torch.Generator(device="cuda").manual_seed(4500)
        self.image= Image.open("a.jpg")
        self.strength=0.2
        self.istep=5
        self.gscale=5.0
        self.prompt=""
        self.neg_prompt=""
        self.i=0
        self.imax=0
        self.image.save("resultats/"+str(self.i)+".png")

    def init_ihm(self):
        self.app=QApplication([])
        self.window=QWidget()
        self.window.setGeometry(1,1,1100,1000)
        self.window.setWindowTitle("stable_diffusion_im2im")
        self.app.setStyle('Fusion')
        self.parametre=QFormLayout()
        self.wstrength=QLineEdit()
        self.wistep=QLineEdit()
        self.wgscale=QLineEdit()
        self.wprompt=QLineEdit()
        self.wantiprompt=QLineEdit()
        self.wstrength.setText(str(self.strength))
        self.wistep.setText(str(self.istep))
        self.wgscale.setText(str(self.gscale))
        self.table=QGridLayout()
        self.parametre.addRow("Prompt",self.wprompt)
        self.parametre.addRow("Anti Prompt",self.wantiprompt)
        self.parametre.addRow("Number of steps",self.wistep)
        self.parametre.addRow("Strength ratio",self.wstrength)
        self.parametre.addRow("Guidance scale",self.wgscale)
        self.params = QWidget()
        self.params.setLayout(self.parametre)
        self.table.addWidget(self.params,0,0,5,2)
        self.llimage=QLabel()
        self.llimage.setBackgroundRole(QPalette.Base)
        self.llimage.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.llimage.setScaledContents(True)
        self.qim = ImageQt(self.image)
        self.pix = QPixmap.fromImage(self.qim)
        self.llimage.setPixmap(self.pix)
        self.limage=QScrollArea()
        self.limage.setWidget(self.llimage)
        self.table.addWidget(self.limage,3,0,6,2)
        self.prec=QPushButton('Back')
        self.psuiv=QPushButton('Back to Next')
        self.suiv=QPushButton('Next')
        self.suiv10=QPushButton('Next x 10')
        self.suiv100=QPushButton('Next x 100')
        self.fichiers=QPushButton('By Files')
        self.suiv.clicked.connect(self.suivant)
        self.suiv10.clicked.connect(self.suivant10)
        self.suiv100.clicked.connect(self.suivant100)
        self.psuiv.clicked.connect(self.psuivant)
        self.prec.clicked.connect(self.precedent)
        self.fichiers.clicked.connect(self.gen_by_fichier)
        self.hlign=QHBoxLayout()
        self.hlign.addWidget(self.prec)
        self.hlign.addWidget(self.psuiv)
        self.hlign.addWidget(self.suiv)
        self.hlign.addWidget(self.suiv10)
        self.hlign.addWidget(self.suiv100)
        self.hlign.addWidget(self.fichiers)
        self.whlign=QWidget()
        self.whlign.setLayout(self.hlign)
        self.table.addWidget(self.whlign,2,0,1,2)
        self.bsave=QPushButton('Sauver')
        self.bload=QPushButton('Loader')
        self.bload.clicked.connect(self.getfiles_load)
        self.bsave.clicked.connect(self.getfiles_save)
        self.hlign2=QHBoxLayout()
        self.hlign2.addWidget(self.bsave)
        self.hlign2.addWidget(self.bload)
        self.whlign2=QWidget()
        self.whlign2.setLayout(self.hlign2)
        self.table.addWidget(self.whlign2,10,0,1,2)
        self.window.setLayout(self.table)

    def getfiles_load(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            self.image= Image.open(filenames[0])
            h, w = self.image.size
            self.llimage.resize(h, w)
            self.qim = ImageQt(self.image)
            self.pix = QPixmap.fromImage(self.qim)
            self.llimage.setPixmap(self.pix)
            self.i=0
            self.imax=0
            self.image.save("resultats/"+str(self.i)+".png")

    def getfiles_save(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
            self.image.save(filenames[0])

    def mise_a_jour(self):
        self.strength=float(self.wstrength.text())
        self.istep=int(self.wistep.text())
        self.gscale=float(self.wgscale.text())
        self.prompt=self.wprompt.text()
        self.neg_prompt=self.wantiprompt.text()

    def suivant(self):
        self.i=self.i+1
        if self.i>self.imax:
            self.imax=self.i
        self.imax
        self.mise_a_jour()
        self.faire_image()
        self.image.save("resultats/"+str(self.i)+".png")
        self.qim = ImageQt(self.image)
        self.pix = QPixmap.fromImage(self.qim)
        self.llimage.setPixmap(self.pix)

    def suivant10(self):
        for j in range(10):
            self.suivant()

    def suivant100(self):
        for j in range(100):
            self.suivant()

    def gen_by_fichier(self):
        wp=self.wprompt.text()
        wap=self.wantiprompt.text()
        fp = open(self.wprompt.text(), 'r')
        fram_p = fp.readlines()
        fp.close()
        fap = open(self.wantiprompt.text(), 'r')
        fram_ap = fap.readlines()
        fap.close()
        for i in range(len(fram_p)):
            self.wprompt.setText(fram_p[i])
            self.wantiprompt.setText(fram_ap[i])
            self.suivant()
        self.wprompt.setText(wp)
        self.wantiprompt.setText(wap)

    def precedent(self):
        if self.i>0:
            self.i=self.i-1
            self.image= Image.open("resultats/"+str(self.i)+".png")
            self.qim = ImageQt(self.image)
            self.pix = QPixmap.fromImage(self.qim)
            self.llimage.setPixmap(self.pix)

    def psuivant(self):
        if self.i<self.imax:
            self.i=self.i+1
            self.image= Image.open("resultats/"+str(self.i)+".png")
            self.qim = ImageQt(self.image)
            self.pix = QPixmap.fromImage(self.qim)
            self.llimage.setPixmap(self.pix)

    def faire_image(self):
        self.neg_embeds = self.compel.build_conditioning_tensor(self.neg_prompt)
        self.embeds = self.compel.build_conditioning_tensor(self.prompt)
        self.image = self.pipe(prompt_embeds=self.embeds,negative_prompt_embeds=self.neg_embeds,image=self.image,strength=self.strength,num_inference_steps=self.istep,guidance_scale=self.gscale,generator=self.generator).images[0]



gg = LogIm2im(sys.argv[1]) # nom fichier model
gg.window.show()
sys.exit(gg.app.exec_())
