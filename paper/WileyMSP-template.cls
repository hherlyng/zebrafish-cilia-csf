\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{WileyMSP-template}

\LoadClass[11pt]{article}
\RequirePackage{graphicx}
\RequirePackage[font=small]{caption}
\RequirePackage[left=2cm,%
                right=2cm,%
                top=2.25cm,%
                bottom=2.25cm,%
                headheight=12pt,%
                a4paper]{geometry}%
\RequirePackage{gensymb}
\RequirePackage{siunitx} % \num{} to format powers of 10
\RequirePackage{float} % Fiure placement
\RequirePackage{amsmath} % Math
\RequirePackage{amssymb} % Blackboard letters
\RequirePackage{booktabs} % Tables
\RequirePackage{color} % Text color
\RequirePackage{bm} % Bold math
\RequirePackage{authblk}
\RequirePackage{stmaryrd} % Double brackets for jump operator
\RequirePackage[colorlinks=true, allcolors=blue]{hyperref} % Links
\RequirePackage[nameinlink, capitalise]{cleveref} % Cross referencing
\RequirePackage[superscript,ref]{cite} % Citation
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

% Author block
\setlength{\affilsep}{1.5em}
\renewcommand\Affilfont{\small} 

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

% Math
\newcommand{\cbar}{\overline{c}} % Mean concentration
\newcommand{\pdifft}[1]{\frac{\partial  #1}{\partial t}} % Time derivative

% Bold faced math characters
\newcommand{\nn}{\bm{n}}
\newcommand{\rr}{\bm{r}}
\newcommand{\uu}{\bm{u}}
\newcommand{\vv}{\bm{v}}
\newcommand{\xx}{\bm{x}}
\newcommand{\JJ}{\bm{J}}
\newcommand{\VV}{\bm{V}}
\newcommand{\bsig}{\bm{\sigma}} % Bold sigma (stress tensor)
\newcommand{\bsigpar}{\hat{\bsig}_{\parallel}} % Tangential traction
\newcommand{\bsigperp}{\hat{\bsig}_{\perp}} % Normal traction
\newcommand{\beps}{\bm{\varepsilon}} % Bold epsilon (symmetric gradient)
\newcommand{\btau}{\bm{\tau}} % Bold tau (cilia traction)

% Discontinuity operators
\newcommand{\avg}[1]{\{#1\}}
\newcommand{\jump}[1]{\llbracket#1\rrbracket}

% Boundaries
\newcommand{\Gs}{\Gamma_{\mathrm{s}}} % Slip boundary
\newcommand{\Gc}{\Gamma_{\mathrm{c}}} % Cilia boundary
\newcommand{\Gp}{\Gamma_{\mathrm{p}}} % Anterior/Posterior pressure boundary
\newcommand{\Gnf}{\Gamma_{\mathrm{nf}}} % No-flux boundary
\newcommand{\Gin}{\Gamma_{\mathrm{in}}} % Inflow boundary
\newcommand{\Gout}{\Gamma_{\mathrm{out}}} % Outflow boundary

% Integrals
\newcommand{\dx}{\, \mathrm d\bm{x}}
\newcommand{\intO}[1]{\int_{\Omega}#1 \, \mathrm d\bm{x}}
\newcommand{\intG}[1]{\int_{\Gamma}#1 \, \mathrm ds}
\newcommand{\intGin}[1]{\int_{\Gamma_{\mathrm{in}}}#1 \, \mathrm ds}
\newcommand{\intGout}[1]{\int_{\Gamma_{\mathrm{out}}}#1 \, \mathrm ds}
\newcommand{\intGc}[1]{\int_{\Gamma_{\mathrm{c}}}#1 \, \mathrm ds}
\newcommand{\intGp}[1]{\int_{\Gamma_{\mathrm{p}}}#1 \, \mathrm ds}

% Norms
\newcommand{\normltwo}[1]{{\vert\vert#1\vert\vert}_{L^2}}
\newcommand{\normltwovec}[1]{{\vert\vert#1\vert\vert}_{\mathbf{L}^2}}
\newcommand{\normlinf}[1]{{\vert\vert#1\vert\vert}_{L^{\infty}}}
\newcommand{\Hnorm}[1]{\vert#1\vert_{\mathbf{H}^1}}
