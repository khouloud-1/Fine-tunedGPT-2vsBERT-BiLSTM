<html xmlns:v="urn:schemas-microsoft-com:vml"
xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns:dt="uuid:C2F41010-65B3-11d1-A29F-00AA00C14882"
xmlns:m="http://schemas.microsoft.com/office/2004/12/omml"
xmlns="http://www.w3.org/TR/REC-html40">

<head>
<meta http-equiv=Content-Type content="text/html; charset=windows-1252">
<meta name=ProgId content=Word.Document>
<meta name=Generator content="Microsoft Word 15">
<meta name=Originator content="Microsoft Word 15">
<link rel=File-List href="index_files/filelist.xml">
<link rel=dataStoreItem href="index_files/item0001.xml"
target="index_files/props002.xml">
<link rel=themeData href="index_files/themedata.thmx">
<link rel=colorSchemeMapping href="index_files/colorschememapping.xml">
</head>

<body lang=EN-US link="#0563C1" vlink="#954F72" style='tab-interval:.5in;
word-wrap:break-word'>

<div class=WordSection1>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span style='font-size:16.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span style='font-size:16.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>HTC: Fine-tuned GPT2 vs BERT-BiLSTM<o:p></o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'>This
page allows you to run two hierarchical text classifiers:<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><i><u><span style='font-size:12.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>1. Fine-tuned GPT2: You need to download the
following files<o:p></o:p></span></u></i></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><a href="HTC_Fine-tunedGPT2.py"><span style='mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>HTC_Fine-tunedGPT2.py</span></a><span
style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><a href="train_40k_Adapted.csv"><span style='mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>train_40k_Adapted.csv</span></a><span
style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><i><u><span style='font-size:12.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>2. BERT-BiLSTM: You need to download the
following files <o:p></o:p></span></u></i></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><a href="HTC_BERT-BiLSTM.py"><span lang=FR style='mso-bidi-font-family:
Calibri;mso-bidi-theme-font:minor-latin;mso-ansi-language:FR'>HTC_BERT-BiLSTM.py</span></a><span
lang=FR style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
mso-ansi-language:FR'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><a href="train_40k_Adapted.csv"><span lang=FR style='mso-bidi-font-family:
Calibri;mso-bidi-theme-font:minor-latin;mso-ansi-language:FR'>train_40k_Adapted.csv</span></a><span
lang=FR style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin;
mso-ansi-language:FR'><o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span lang=FR style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:
minor-latin;mso-ansi-language:FR'><o:p>&nbsp;</o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span lang=FR style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:
minor-latin;mso-ansi-language:FR'><o:p>&nbsp;</o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span style='font-size:14.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>Before running the two classifiers, Install
Necessary Libraries<o:p></o:p></span></b></p>

<p class=MsoCommentText style='margin-bottom:0in'>pip install pandas</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install numpy</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install nltk</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install scikit-learn</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install tensorflow</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install transformers</p>
<p class=MsoCommentText style='margin-bottom:0in'>pip install torch</p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'>If
you use material from this page, please cite it as follows:<o:p></o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;line-height:normal;layout-grid-mode:
char;mso-layout-grid-align:none'><span style='font-family:"Times New Roman",serif;
mso-fareast-font-family:"Times New Roman";mso-font-kerning:0pt;mso-ligatures:
none;mso-fareast-language:EN-GB'>Djelloul BOUCHIHA, Abdelghani BOUZIANE,
Noureddine DOUMI, Benamar HAMZAOUI, and Hacene-Sofiane BOUKLI… (still under
review)<o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><b><span style='font-size:14.0pt;mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>Contact<o:p></o:p></span></b></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'>For
help, don’t hesitate to contact me: </span><a
href="mailto:bouchiha@cuniv-naama.dz"><span style='mso-bidi-font-family:Calibri;
mso-bidi-theme-font:minor-latin'>bouchiha@cuniv-naama.dz</span></a><span
style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'> <o:p></o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

<p class=MsoNormal style='margin-bottom:0in;text-align:justify;line-height:
normal'><span style='mso-bidi-font-family:Calibri;mso-bidi-theme-font:minor-latin'><o:p>&nbsp;</o:p></span></p>

</div>

</body>

</html>
