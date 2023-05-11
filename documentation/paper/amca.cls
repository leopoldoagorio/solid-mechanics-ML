%% This program is  free software; you can redistribute  it and/or modify
%% it under the  terms of the GNU General Public  License as published by
%% the Free Software Foundation; either  version 2 of the License, or (at
%% your option) any later version.

%% This program  is distributed in the  hope that it will  be useful, but
%% WITHOUT   ANY  WARRANTY;   without  even   the  implied   warranty  of
%% MERCHANTABILITY  or FITNESS  FOR A  PARTICULAR PURPOSE.   See  the GNU
%% General Public License for more details.

%% You  should have received  a copy  of the  GNU General  Public License
%% along  with  this  program;  if   not,  write  to  the  Free  Software
%% Foundation, Inc.,  59 Temple Place, Suite 330,  Boston, MA 02111-1307,
%% USA.

%% This is file `amca.cls', the class file for publications
%% of AMCA (Asociacion Argentina de Mecanica Computacional).
%% http://www.amcaonline.org.ar
%% This file is maintained by the AMCA Publication Style Committee.
%% For support contact see http://www.amcaonline.org.ar/
%% 
%% (originally written by Ton van den Boogaard for WCCM IV).
%% 
%% $Id texstuff-1.0.10-18-g74ee5ff Sat Jun 9 23:27:40 2007 -0300$
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{amca}[2002/06/10 AMCA paper class]
%
\DeclareOption{10pt}{\OptionNotUsed}
\DeclareOption{11pt}{\OptionNotUsed}
\DeclareOption{twocolumn}{\OptionNotUsed}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
%
\def\amcalanguage{1}
\DeclareOption{spanish}{
  \def\amcalanguage{0}
}
\def\amcapagenumbers{0}
\DeclareOption{pagenumbers}{
  \def\amcapagenumbers{1}
}
%
\DeclareOption{english}{
  \def\amcalanguage{1}
}
%
\DeclareOption{portuguese}{
  \def\amcalanguage{2}
}
%
\def\amcauselinks{1}
\DeclareOption{links}{
  \def\amcauselinks{1}
}
\DeclareOption{nolinks}{
  \def\amcauselinks{0}
}
%
\ProcessOptions\relax
%
\LoadClass[12pt]{article}
%
\RequirePackage{calc}
\RequirePackage{indentfirst}
%
% This allows that copying from PDF files preserves accents. 
\RequirePackage{times}
\RequirePackage[T1]{fontenc} % Usar T1 enconding
%
%% Title
%
\def\englishtitle#1{\gdef\@englishtitle{#1}}
\def\@englishtitle{}
%
\renewcommand{\@maketitle}{%
  \newpage
  \null
  \vspace{2cm}%
  {\begin{center}%
     {\fontsize{14}{16}\selectfont \bfseries \MakeUppercase{\@title} \par}%
     \vspace{12pt}% AMCA
     \if!\@englishtitle!\else
      {\fontsize{12}{14}\selectfont \bfseries \MakeUppercase{\@englishtitle}  
       \vspace{12pt}
       \par
       }
    \fi
    {
        \lineskip .5em%
        {\setlength{\tabcolsep}{0in}
        \begin{tabular}[t]{c}%
          \bfseries \@author
        \end{tabular}\par} }%
  \end{center}}% end of \parbox
  } % end of \@maketitle

\renewenvironment{abstract}{%
      \noindent \bfseries \abstractname.%
      \mdseries\small%
}
%\newenvironment{engabstract}{%
%     \noindent \bfseries \abstractname.%
%     \mdseries\small%
%}
{\thispagestyle{empty}%
\pagebreak}
%
\newenvironment{keywords}
{\noindent\textbf{\keywordsname: }}
{\vspace{12pt}}
%
%\newenvironment{engkeywords}
%{\noindent\textbf{\keywordsname: }}
%{\vspace{12pt}}
%
\renewenvironment{thebibliography}[1]
     {\section*{\refname
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \parsep 0pt
            \itemsep 0pt
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
%
\renewcommand{\thesubsection}   {\textup{\thesection.\arabic{subsection}}}
\renewcommand{\thesubsubsection}{\textup{%
    \thesubsection .\arabic{subsubsection}}}
%
\renewcommand{\section}{\@startsection{section}{1}{0mm}%
                                     {-12pt}%
                                     {6pt}%
                                     {\bfseries\normalsize\MakeUppercase}}%
\renewcommand{\subsection}{\@startsection{subsection}{2}{0mm}%
                                     {-12pt}%
                                     {6pt}%
                                     {\bfseries\normalsize}}%
%
% Previously in class `mecom'
%\setlength{\oddsidemargin}{0mm}
%\setlength{\evensidemargin}{0mm}
%\setlength{\textheight}{210mm}
%\setlength{\textwidth}{160mm}
\setlength{\textfloatsep}{12pt}
\setlength{\abovedisplayskip}{6pt}
\setlength{\belowdisplayskip}{6pt}
\setlength{\parindent}{5mm}
\setlength{\abovecaptionskip}{6pt}
\setlength{\belowcaptionskip}{6pt}
%
% Previously in `instructions.tex'
\setlength{\textwidth}{16cm}
\setlength{\textheight}{24cm}
\setlength{\oddsidemargin}{-0.04cm}
\setlength{\topmargin}{8.1mm}
\setlength{\headheight}{0mm}
\setlength{\headsep}{0mm}
%\setlength{\affilsep}{0pt}
%
\renewcommand{\@makecaption}[2]{%
  \vspace{\abovecaptionskip}%
  {\fontsize{10}{12}\selectfont
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi}
  \vskip\belowcaptionskip}
%
\def\voidaffil{$\mskip 0mu$}
%
\RequirePackage[noblocks]{authblk}
\renewcommand\AB@affilsepx{\protect\vspace{12pt}\\\protect\Affilfont }
\renewcommand\Authands{ and }
\renewcommand{\Affilfont}{\mdseries\itshape\small}
\renewcommand{\Authfont}{\bfseries}
%
\RequirePackage{natbib}
\bibliographystyle{amca}
%\bibliographystyle{unsrt}
\bibpunct{(}{)}{;}{a}{,}{,}
\setlength{\bibsep}{0pt}
%
\ifnum \amcapagenumbers = 0
\pagestyle{empty}
\fi
%
%
\ifnum \amcalanguage = 0
  \renewcommand\Authand{ y }%
  \renewcommand\Authands{ y }
  \RequirePackage[spanish]{babel}
  \def\keywordsname{Palabras clave}
  \def\amcabtxand{y}
  \def\amcabtxeditors{editores}
  \def\amcabtxeditedby{editado por}
  \def\amcabtxedition{edici\'on}
  \def\amcabtxvolume{volumen}
  \def\amcabtxof{de}
  \def\amcabtxnumber{n\'umero}
  \def\amcabtxpages{p\'aginas}
  \def\amcabtxchapter{cap\'\i{}tulo}
  \def\amcabtxtechrep{Informe T\'ecnico}
  \def\amcabtxmaster{Tesis de Maestr\'\i{}a}
  \def\amcabtxphd{Tesis de Doctorado}
  \def\amcabtxpage{p\'agina}
  \def\amcabtxin{En}
  \addto\captionsspanish{%
     \def\tablename{Tabla}}%
\fi
%
\ifnum \amcalanguage = 1
  \renewcommand\Authand{ and }%
  \renewcommand\Authands{ and }
  \RequirePackage[english]{babel}
  \def\keywordsname{Keywords}
  \def\amcabtxand{and}
  \def\amcabtxeditors{editors}
  \def\amcabtxeditedby{edited by}
  \def\amcabtxedition{edition}
  \def\amcabtxvolume{volume}
  \def\amcabtxof{of}
  \def\amcabtxnumber{number}
  \def\amcabtxpages{pages}
  \def\amcabtxchapter{chapter}
  \def\amcabtxtechrep{Technical Report}
  \def\amcabtxmaster{Master's Thesis}
  \def\amcabtxphd{Ph.D. thesis}
  \def\amcabtxpage{page}
  \def\amcabtxin{In}
\fi
%
\ifnum \amcalanguage = 2
  \renewcommand\Authand{ e }%
  \renewcommand\Authands{ e }
  \RequirePackage[portuguese]{babel}
  \def\keywordsname{Palavras-chave}
  \def\amcabtxand{e}
  \def\amcabtxeditors{editores}
  \def\amcabtxeditors{editores}
  \def\amcabtxeditedby{editado por}
  \def\amcabtxedition{edi\c c\~ao}
  \def\amcabtxvolume{volume}
  \def\amcabtxof{de}
  \def\amcabtxnumber{n\'umero}
  \def\amcabtxpages{p\'aginas}
  \def\amcabtxchapter{cap\'\i{}tulo}
  \def\amcabtxtechrep{Relat\'orio T\'ecnico}
  \def\amcabtxmaster{Tesis de Mestrado}
  \def\amcabtxphd{Tesis de Doutorado}
  \def\amcabtxpage{pagina}
  \def\amcabtxin{Em}
\fi
%
\ifnum \amcauselinks = 1
% Use color links
\RequirePackage{color}
\definecolor{url}{rgb}{0.8,0.,0.}
\RequirePackage[colorlinks,linkcolor=blue,%
                urlcolor=url,citecolor=blue,%
                plainpages=false,pdfpagelabels]{hyperref}
\RequirePackage{nameref,url}
\def\UrlFont{\mdseries}
\fi
