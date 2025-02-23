\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{WileyMSP-template}

\LoadClass{article}
\RequirePackage{ragged2e}
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage[margin=0.5in]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{gensymb}
\RequirePackage{siunitx} % \num{} to format powers of 10
\RequirePackage{float} % Fiure placement
\RequirePackage{amsmath} % Math
\RequirePackage{amssymb} % Blackboard letters
\RequirePackage{booktabs} % Tables
\RequirePackage{color} % Text color
\RequirePackage{bm} % Bold math
\RequirePackage{stmaryrd} % Double brackets for jump operator
\RequirePackage[colorlinks=true, allcolors=blue]{hyperref} % Links
\RequirePackage[nameinlink, capitalise]{cleveref} % Cross referencing
\RequirePackage[superscript,ref]{cite} % Citation
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\RaggedRight

\renewcommand{\maketitle}{
\huge
\@title
\medskip
\global\@topnum\z@
   }
   
 \renewenvironment{author} {
 \large
 \emph
  }
 

\newcommand{\keywords}[1]{
\medskip
Keywords: \textit{#1}
}

\newcommand{\dedication}[1]{
\medskip
\textit{#1}
}

\newenvironment{affiliations}{
\medskip
\large
}

\renewenvironment{abstract}{
\small
\medskip\medskip
}

\newcommand{\threesubsection}[1]{
\medskip
\textit{#1}: 
}

% Bracket citations
\makeatletter
\renewcommand\@citess[1]{\textsuperscript{[#1]}} 
\makeatother

% Add panels to figure references
\newcommand{\Crefpanels}[2]{\hyperref[#1]{\Cref*{#1}#2}}

%% -----  Some custom commands  ----- %%

%% EDITING: REMOVE THESE BEFORE SUBMISSION
\RequirePackage{xcolor}
\newcommand{\fixme}[1]{\textcolor{red}{#1}}
\newcommand{\lyng}[1]{\textcolor{blue}{#1}}
\newcommand{\mer}[1]{\textcolor{magenta}{#1}}
\newcommand{\mk}[1]{\textcolor{olive}{MK: #1}}

% Citation
\newcommand{\newcite}[1]{\textsuperscript{\cite{#1}}}

% Math
\newcommand{\cbar}{\overline{c}} % Mean concentration
\newcommand{\pdifft}[1]{\frac{\partial  #1}{\partial t}}

% Bold faced math characters
\newcommand{\ff}{\bm{f}}
\newcommand{\nn}{\bm{n}}
\newcommand{\rr}{\bm{r}}
\newcommand{\uu}{\bm{u}}
\newcommand{\vv}{\bm{v}}
\newcommand{\xx}{\bm{x}}
\newcommand{\JJ}{\bm{J}}
\newcommand{\VV}{\bm{V}}
\newcommand{\bsig}{\bm{\sigma}}
\newcommand{\bsigpar}{\hat{\bsig}_{\parallel}}
\newcommand{\bsigperp}{\hat{\bsig}_{\perp}}
\newcommand{\beps}{\bm{\varepsilon}}
\newcommand{\btau}{\bm{\tau}}

% Discontinuity operators
\newcommand{\avg}[1]{\{#1\}}
\newcommand{\jump}[1]{\llbracket#1\rrbracket}

% Boundaries
\newcommand{\Gs}{\Gamma_{\mathrm{s}}}
\newcommand{\Gc}{\Gamma_{\mathrm{c}}}
\newcommand{\Gp}{\Gamma_{\mathrm{p}}}
\newcommand{\Gnf}{\Gamma_{\mathrm{nf}}}
\newcommand{\Gin}{\Gamma_{\mathrm{in}}}
\newcommand{\Gout}{\Gamma_{\mathrm{out}}}

% Integrals
\newcommand{\dx}{\, \mathrm d\bm{x}}
\newcommand{\intO}[1]{\int_{\Omega}#1 \, \mathrm d\bm{x}}
\newcommand{\intG}[1]{\int_{\Gamma}#1 \, \mathrm ds}
\newcommand{\intF}[1]{\int_{\mathcal{F}_k}#1 \, \mathrm dS}
\newcommand{\intGin}[1]{\int_{\Gamma_{\mathrm{in}}}#1 \, \mathrm ds}
\newcommand{\intGout}[1]{\int_{\Gamma_{\mathrm{out}}}#1 \, \mathrm ds}
\newcommand{\intGc}[1]{\int_{\Gamma_{\mathrm{c}}#1 \, \mathrm ds}}
\newcommand{\intGp}[1]{\int_{\Gamma_{\mathrm{p}}#1 \, \mathrm ds}}

% Norms
\newcommand{\normltwo}[1]{{ \vert\vert#1\vert\vert}_{L^2}}
\newcommand{\normltwovec}[1]{{ \vert\vert#1\vert\vert}_{\mathbf{L}^2}}
\newcommand{\normlinf}[1]{{\vert\vert#1\vert\vert}_{L^{\infty}}}
\newcommand{\Hnorm}[1]{\vert\vert#1\vert\vert_{\mathbf{H}^1}}