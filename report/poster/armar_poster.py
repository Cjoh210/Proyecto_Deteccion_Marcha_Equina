"""
Llena la plantilla FIDIA (90 x 120 cm) con el contenido del proyecto, en español e inglés.
Respeta el marco, las posiciones, la tipografía (Arial) y los colores de la plantilla; solo se
reemplazan textos e imágenes de ejemplo.

Uso: python report/poster/armar_poster.py <plantilla.pptx>
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
EMU = 360000  # EMU por cm

TXT = {
    "en": {
        "title": "Equine Lameness Detection with Data Mining, Machine Learning and the Cloud",
        "authors": ["Carlos Josué Barrera Vargas",
                    "Instructors: Iván Felipe Mariño Barreto, Zalma Valentina Moreno Galeano",
                    "Faculty of Data Engineering and Artificial Intelligence"],
        "h": ["Introduction", "Methodology", "Results", "Conclusions", "References"],
        "intro": "Lameness is the most common health problem in horses, and when it is mild, veterinarians often disagree on the diagnosis. Colombia has nearly one million horses. This project builds the analytical foundation of a low-cost system that will detect lameness from a phone video, aligned with SDGs 3 and 8.",
        "method": "Following CRISP-DM, three sources were integrated: scientific papers collected through the OpenAlex API, a catalogue of public datasets and inertial-sensor signals from 22 horses with induced lameness. The pipeline covers cleaning, feature engineering, EDA, PCA, clustering and association rules, ensemble models validated on unseen horses, and a serverless AWS deployment built with Terraform.",
        "results": [("Research gap: ", "computer vision rose from 21 % to 52 % of papers, yet only one study on horse lameness with vision has Colombian authors."),
                    ("Data gap: ", "no public dataset combines horse video with lameness labels."),
                    ("Clinical signal: ", "head and pelvis asymmetry reveal fore- and hindlimb lameness."),
                    ("Model: ", "an optimized Random Forest reached a ROC-AUC of 0.71 on unseen horses."),
                    ("Cloud: ", "the live API returns predictions and stores them in DynamoDB, at zero actual cost.")],
        "concl": [("", "Data mining mapped a field moving towards computer vision and revealed a clear gap in open data and in Latin American research. Gait asymmetry captures lameness in an interpretable way, and the model generalizes to new horses. The whole system runs in the cloud as code at almost no cost."),
                  ("Next steps: ", "estimate the same asymmetry measures from phone video with pose estimation, and validate the system on Colombian criollo horses assessed by veterinarians.")],
        "refs": ["[1] K. G. Keegan et al., “Repeatability of subjective evaluation of lameness in horses,” Equine Vet. J., 2010.",
                 "[2] M. Hammarberg et al., “Rater agreement of visual lameness assessment in horses during lungeing,” Equine Vet. J., 2016.",
                 "[3] I. Smit et al., “Kinematic data of the Icelandic horse with induced lameness at walk, trot and tölt,” DataverseNL, 2025.",
                 "[4] J. Priem, H. Piwowar and R. Orr, “OpenAlex: A fully-open index of scholarly works,” arXiv:2205.01833, 2022.",
                 "[5] A. Mathis et al., “DeepLabCut: markerless pose estimation of user-defined body parts,” Nat. Neurosci., 2018.",
                 "[6] C. Li et al., “The Poses for Equine Research Dataset (PFERD),” Sci. Data, 2024.",
                 "[7] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” NeurIPS, 2017."],
        "qr": ["Code, data and infrastructure as code:", "github.com/Cjoh210/Proyecto_Deteccion_Marcha_Equina"],
        "out": "CarlosBarrera_poster_EN.pptx",
    },
    "es": {
        "title": "Detección de cojera equina con minería de datos, aprendizaje automático y la nube",
        "authors": ["Carlos Josué Barrera Vargas",
                    "Docentes: Iván Felipe Mariño Barreto, Zalma Valentina Moreno Galeano",
                    "Facultad de Ingeniería de Datos e Inteligencia Artificial"],
        "h": ["Introducción", "Metodología", "Resultados", "Conclusiones", "Bibliografía"],
        "intro": "La cojera es el problema de salud más común en los caballos y, cuando es leve, los veterinarios suelen no coincidir en el diagnóstico. Colombia tiene cerca de un millón de equinos. Este proyecto construye la base analítica de un sistema de bajo costo que detectará la cojera desde un video del celular, alineado con los ODS 3 y 8.",
        "method": "Siguiendo CRISP-DM, se integraron tres fuentes: artículos científicos obtenidos con la API de OpenAlex, un catálogo de datasets públicos y señales de sensores inerciales de 22 caballos con cojera inducida. El pipeline incluye limpieza, ingeniería de características, EDA, PCA, clustering y reglas de asociación, modelos de ensamble validados con caballos no vistos y un despliegue sin servidores en AWS construido con Terraform.",
        "results": [("Vacío de investigación: ", "la visión por computador pasó del 21 % al 52 % de los trabajos, pero solo un estudio sobre cojera equina con visión tiene autores en Colombia."),
                    ("Vacío de datos: ", "ningún dataset público combina video de caballos con etiqueta de cojera."),
                    ("Señal clínica: ", "la asimetría de la cabeza y la pelvis revela la cojera anterior y posterior."),
                    ("Modelo: ", "un Random Forest optimizado alcanzó un ROC-AUC de 0,71 con caballos no vistos."),
                    ("Nube: ", "la API en vivo entrega predicciones y las guarda en DynamoDB, con costo real de cero.")],
        "concl": [("", "La minería de datos mapeó un campo que avanza hacia la visión por computador y reveló un vacío claro en datos abiertos y en investigación latinoamericana. La asimetría de la marcha captura la cojera de forma interpretable y el modelo generaliza a caballos nuevos. Todo el sistema funciona en la nube como código, con un costo casi nulo."),
                  ("Siguientes pasos: ", "estimar las mismas medidas de asimetría desde video con estimación de pose y validar el sistema con caballos criollos colombianos evaluados por veterinarios.")],
        "refs": ["[1] K. G. Keegan et al., «Repeatability of subjective evaluation of lameness in horses», Equine Vet. J., 2010.",
                 "[2] M. Hammarberg et al., «Rater agreement of visual lameness assessment in horses during lungeing», Equine Vet. J., 2016.",
                 "[3] I. Smit et al., «Kinematic data of the Icelandic horse with induced lameness at walk, trot and tölt», DataverseNL, 2025.",
                 "[4] J. Priem, H. Piwowar y R. Orr, «OpenAlex: A fully-open index of scholarly works», arXiv:2205.01833, 2022.",
                 "[5] A. Mathis et al., «DeepLabCut: markerless pose estimation of user-defined body parts», Nat. Neurosci., 2018.",
                 "[6] C. Li et al., «The Poses for Equine Research Dataset (PFERD)», Sci. Data, 2024.",
                 "[7] S. M. Lundberg y S.-I. Lee, «A unified approach to interpreting model predictions», NeurIPS, 2017."],
        "qr": ["Código, datos e infraestructura como código:", "github.com/Cjoh210/Proyecto_Deteccion_Marcha_Equina"],
        "out": "CarlosBarrera_poster_ES.pptx",
    },
}

# Posiciones nuevas (cm): x, y, ancho[, alto]
POS = {
    "TextBox 42": (3.0, 18.9, 84.0),
    "TextBox 45": (5.0, 29.7, 33.0),
    "TextBox 46": (42.2, 29.7, 44.0),
    "TextBox 49": (60.3, 60.0, 26.7),
    "TextBox 52": (4.7, 86.3, 41.5),
    "TextBox 55": (51.2, 86.3, 34.5),
}
ARQ = ("arquitectura.png", 4.0, 74.4, 55.3, 6.6)  # franja nueva bajo la gráfica de resultados (cm)
LINEAS = ("lineas_investigacion.png", 4.7, 97.4, 41.5, 16.6)  # mapa de clusters bajo las conclusiones (cm)
SUBIR = 5.4        # la sección 04 y la bibliografía suben este espacio (cm)
SUBEN = ["Group 30", "Group 32", "Group 35", "TextBox 38", "Group 4"]
IMGS = {  # grupo -> (archivo, x, y, w, h) en cm
    "Group 43": ("intro_tendencia.png", 5.0, 39.0, 31.0, 15.0),
    "Group 47": ("metodologia.png", 42.2, 39.0, 40.0, 15.0),
    "Group 50": ("resultados.png", 4.0, 60.0, 55.3, 14.0),
    "Group 53": ("qr_repositorio.png", 51.6, 103.2, 7.4, 7.4),
}


def emu(cm):
    return str(int(round(cm * EMU)))


def rpr(sz, color, bold=False):
    face = "Arial Bold" if bold else "Arial"
    b = ' b="true"' if bold else ""
    return (f'<a:rPr lang="en-US" sz="{sz}"{b}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:latin typeface="{face}"/><a:ea typeface="{face}"/><a:cs typeface="{face}"/><a:sym typeface="{face}"/></a:rPr>')


def para(runs, sz, color, lnspc, algn="l", bullet=False, after=0):
    ppr = f'<a:pPr algn="{algn}"' + (' marL="685800" indent="-685800"' if bullet else "") + ">"
    ppr += f'<a:lnSpc><a:spcPts val="{lnspc}"/></a:lnSpc>'
    if after:
        ppr += f'<a:spcAft><a:spcPts val="{after}"/></a:spcAft>'
    ppr += ('<a:buFont typeface="Arial"/><a:buChar char="•"/>' if bullet else "<a:buNone/>") + "</a:pPr>"
    body = "".join(f'<a:r>{rpr(sz, color, b)}<a:t xml:space="preserve">{escape(t)}</a:t></a:r>' for t, b in runs if t)
    return f"<a:p>{ppr}{body}</a:p>"


def sp_block(xml, name):
    m = re.search(r'<p:sp>(?:(?!</p:sp>).)*?name="%s".*?</p:sp>' % re.escape(name), xml, re.S)
    assert m, name
    return m


def grp_block(xml, name):
    m = re.search(r'<p:grpSp>(?:(?!<p:grpSp>).)*?name="%s".*?</p:grpSp>' % re.escape(name), xml, re.S)
    assert m, name
    return m


def set_paras(xml, name, paras):
    m = sp_block(xml, name)
    b = m.group(0)
    b = re.sub(r"(<a:lstStyle/>).*(</p:txBody>)", lambda mm: mm.group(1) + "".join(paras) + mm.group(2), b, flags=re.S)
    return xml[:m.start()] + b + xml[m.end():]


def set_simple(xml, name, text):
    """Reemplaza el texto conservando el formato del primer run (títulos y números)."""
    m = sp_block(xml, name)
    b = m.group(0)
    b = re.sub(r"<a:t>[^<]*</a:t>", f"<a:t>{escape(text)}</a:t>", b, count=1)
    return xml[:m.start()] + b + xml[m.end():]


def move(xml, name, x, y, w, h=None):
    m = sp_block(xml, name)
    b = m.group(0)
    def rep(mm):
        hh = emu(h) if h else mm.group(2)
        return f'<a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{hh}"/>'
    b = re.sub(r'<a:off x="\d+" y="\d+"/><a:ext cx="\d+" cy="(\d+)"/>'.replace('cy="(\\d+)"', 'cy="(\\d+)"'),
               lambda mm: f'<a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h) if h else mm.group(1)}"/>', b, count=1)
    return xml[:m.start()] + b + xml[m.end():]


def set_image(xml, group, rid, x, y, w, h):
    m = grp_block(xml, group)
    b = m.group(0)
    W, H = emu(w), emu(h)
    b = re.sub(r'<a:off x="\d+" y="\d+"/><a:ext cx="\d+" cy="\d+"/><a:chOff x="0" y="0"/><a:chExt cx="\d+" cy="\d+"/>',
               f'<a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{W}" cy="{H}"/><a:chOff x="0" y="0"/><a:chExt cx="{W}" cy="{H}"/>', b, count=1)
    b = re.sub(r'(<p:sp>.*?<a:off x="0" y="0"/>)<a:ext cx="\d+" cy="\d+"/>', lambda mm: mm.group(1) + f'<a:ext cx="{W}" cy="{H}"/>', b, count=1, flags=re.S)
    b = re.sub(r'<a:path h="\d+" w="\d+">.*?</a:path>',
               f'<a:path h="{H}" w="{W}"><a:moveTo><a:pt x="0" y="0"/></a:moveTo><a:lnTo><a:pt x="{W}" y="0"/></a:lnTo>'
               f'<a:lnTo><a:pt x="{W}" y="{H}"/></a:lnTo><a:lnTo><a:pt x="0" y="{H}"/></a:lnTo><a:lnTo><a:pt x="0" y="0"/></a:lnTo><a:close/></a:path>',
               b, count=1, flags=re.S)
    b = re.sub(r'r:embed="rId\d+"', f'r:embed="{rid}"', b, count=1)
    b = re.sub(r"<a:fillRect[^>]*/>", '<a:fillRect l="0" t="0" r="0" b="0"/>', b, count=1)
    return xml[:m.start()] + b + xml[m.end():]


def add_textbox(xml, sid, name, x, y, w, paras):
    sp = (f'<p:sp><p:nvSpPr><p:cNvPr name="{name}" id="{sid}"/><p:cNvSpPr txBox="true"/><p:nvPr/></p:nvSpPr>'
          f'<p:spPr><a:xfrm rot="0"><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(2)}"/></a:xfrm>'
          f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr anchor="t" rtlCol="false" tIns="0" lIns="0" bIns="0" rIns="0">'
          f'<a:spAutoFit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')
    return xml.replace("</p:spTree>", sp + "</p:spTree>")


def subir(xml, name, dy, crecer=False):
    """Sube un elemento de primer nivel dy cm; si crecer, alarga su alto lo mismo para conservar el borde inferior."""
    m = re.search(r'<p:(sp|grpSp)>(?:(?!</p:\1>).)*?name="%s".*?</p:\1>' % re.escape(name), xml, re.S)
    assert m, name
    b = m.group(0)
    def rep(mm):
        y = int(mm.group(2)) - int(round(dy * EMU))
        cy = int(mm.group(4)) + (int(round(dy * EMU)) if crecer else 0)
        return f'<a:off x="{mm.group(1)}" y="{y}"/><a:ext cx="{mm.group(3)}" cy="{cy}"/>'
    b = re.sub(r'<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/>', rep, b, count=1)
    return xml[:m.start()] + b + xml[m.end():]


def add_picture(xml, sid, name, rid, x, y, w, h):
    pic = (f'<p:pic><p:nvPicPr><p:cNvPr id="{sid}" name="{name}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
           f'<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>'
           f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
           f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>')
    return xml.replace("</p:spTree>", pic + "</p:spTree>")


def build(template, lang):
    t = TXT[lang]
    work = HERE / f"_build_{lang}"
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(template) as z:
        z.extractall(work)
    slide = work / "ppt/slides/slide1.xml"
    rels = work / "ppt/slides/_rels/slide1.xml.rels"
    xml, rx = slide.read_text(encoding="utf-8"), rels.read_text(encoding="utf-8")

    # imágenes nuevas
    for i, (grp, (fname, x, y, w, h)) in enumerate(IMGS.items()):
        rid, media = f"rId{20 + i}", f"poster_{i + 1}.png"
        shutil.copy(HERE / "figures" / lang / fname, work / "ppt/media" / media)
        rx = rx.replace("</Relationships>", f'<Relationship Id="{rid}" Target="../media/{media}" '
                        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/></Relationships>')
        xml = set_image(xml, grp, rid, x, y, w, h)

    # arquitectura en AWS, franja bajo los resultados
    shutil.copy(HERE / "figures" / lang / ARQ[0], work / "ppt/media/poster_arquitectura.png")
    rx = rx.replace("</Relationships>", '<Relationship Id="rId30" Target="../media/poster_arquitectura.png" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/></Relationships>')
    xml = add_picture(xml, 91, "Arquitectura AWS", "rId30", *ARQ[1:])
    # mapa de las líneas de investigación bajo las conclusiones
    shutil.copy(HERE / "figures" / lang / LINEAS[0], work / "ppt/media/poster_lineas.png")
    rx = rx.replace("</Relationships>", '<Relationship Id="rId31" Target="../media/poster_lineas.png" '
                    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"/></Relationships>')
    xml = add_picture(xml, 92, "Lineas de investigacion", "rId31", *LINEAS[1:])

    # títulos de sección y numeración (05 -> 04)
    for name, txt in zip(["TextBox 13", "TextBox 21", "TextBox 29", "TextBox 37", "TextBox 38"], t["h"]):
        xml = set_simple(xml, name, txt)
    xml = set_simple(xml, "TextBox 34", "04")
    xml = set_simple(xml, "TextBox 41", t["title"])

    # autores
    xml = set_paras(xml, "TextBox 42", [para([(a, i == 0)], 3399, "444444", 4079, algn="ctr") for i, a in enumerate(t["authors"])])
    # cuerpo
    xml = set_paras(xml, "TextBox 45", [para([(t["intro"], False)], 3000, "1F1F1F", 3960)])
    xml = set_paras(xml, "TextBox 46", [para([(t["method"], False)], 3000, "1F1F1F", 3960)])
    xml = set_paras(xml, "TextBox 49", [para([(k, True), (v, False)], 3000, "1F1F1F", 3960, bullet=True, after=1400) for k, v in t["results"]])
    xml = set_paras(xml, "TextBox 52", [para([(k, True), (v, False)], 3000, "1F1F1F", 3960, after=2000) for k, v in t["concl"]])
    xml = set_paras(xml, "TextBox 55", [para([(r, False)], 1899, "444444", 2462, after=500) for r in t["refs"]])
    for name, pos in POS.items():
        xml = move(xml, name, *pos)
    for name in SUBEN:
        xml = subir(xml, name, SUBIR, crecer=(name == "Group 4"))
    # texto junto al QR
    xml = add_textbox(xml, 90, "TextBox QR", 60.3, 104.6, 25.5,
                      [para([(t["qr"][0], True)], 2200, "1F1F1F", 2900, after=300), para([(t["qr"][1], False)], 2000, "004AAD", 2700)])

    slide.write_text(xml, encoding="utf-8"); rels.write_text(rx, encoding="utf-8")
    out = HERE / t["out"]
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    print("ok", out.name)


if __name__ == "__main__":
    for lang in TXT:
        build(sys.argv[1], lang)
