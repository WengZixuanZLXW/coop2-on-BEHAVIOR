"""One style for every figure.

Fonts are points, geometry is inches, so a mark or a label is the same size
in every figure regardless of how a panel is laid out; a figure is composed
at the paper's text width and included unscaled. The palette is the paper's:
task tools and events warm, environment tools and events blue, information
tools and messages white, activations gray, the T / D / Z bands tinted."""

# fonts (points); FONT_MIN is the floor
FONT_TITLE = 8.0
FONT_LABEL = 7.0
FONT_NODE = 6.5
FONT_MIN = 6.0

# bands
BAND_T, BAND_D, BAND_Z = "#fdf2e4", "#eef0f4", "#e3ebf6"

# tools (squares) and events (circles)
TASK_TOOL, ENV_TOOL, COMM_TOOL = "#e8574a", "#5b7fbf", "#ffffff"
TASK_EVENT, ENV_EVENT, COMM_EVENT = "#f0a04b", "#2f4f8f", "#ffffff"

# activations and agents
ACT_FILL, ACT_EDGE = "#e2e6ee", "#9aa3b2"
AGENT_FILL, AGENT_EDGE = "#e3ebf6", "#35578a"

# lines
INK = "#222222"
MARK_EDGE = "#555555"
GRAY = "#9a9a9a"
CLOSED = "#7f7f7f"     # a closed task line: its versions, actions, and labels
FAINT = "#cfcfcf"
SEND = "#4a5a75"
CONTEXT = "#b8b8b8"
CONDITIONAL = "#8a4b35"
LOOP = "#a03030"
EVIDENCE = "#7a4c9a"
LABEL_RED = "#c0392b"
OK_GREEN = "#3a7d44"

# marks (inches)
NODE_R = 0.11          # agent node radius (a pill widens for long names)
SQ = 0.05              # tool square side
R_EVENT = 0.03         # event circle radius
R_TASK = 0.032         # task version radius in the T band
T_SQ = 0.036           # task action square side in the T band (D's calls keep SQ)

TEXT_WIDTH_IN = 5.5    # the paper's text width; full-width figures are composed to it
