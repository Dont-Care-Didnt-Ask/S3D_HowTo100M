# Building prompts for Franka Kitchen, with some variations for each prompt
articles = ("A", "The")
names = ("robotic arm", "robot", "manipulator")

FRANKA_BASELINE = ["A robotic arm in the kitchen"]

BOTTOMKNOB = [f"{article} {name} {verb} the knob which activates the bottom burner  in the kitchen" 
    for article in articles
    for name in names
    for verb in ("twists", "is twisting")
]

TOPKNOB = [f"{article} {name} {verb} the knob which activates the top burner  in the kitchen" 
    for article in articles
    for name in names
    for verb in ("twists", "is twisting")
]

LIGHTSWITCH = [f"{article} {name} {verb} the light switch in the kitchen" 
    for article in articles
    for name in names
    for verb in ("turns on", "is turning on")
]

HINGE = [f"{article} {name} {verb} the left hinge door in the kitchen"
    for article in articles
    for name in names
    for verb in ("opens", "is opening")
]

SLIDE = [f"{article} {name} {verb} the slide door in the kitchen"
    for article in articles
    for name in names
    for verb in ("opens", "is opening")
]

MICROWAVE = [f"{article} {name} {verb} the microwave door in the kitchen"
    for article in articles
    for name in names
    for verb in ("opens", "is opening")
]

KETTLE = [f"{article} {name} {verb} the kettle to the top left burner in the kitchen"
    for article in articles
    for name in names
    for verb in ("moves", "is moving")
]

FRANKA_PROMPT_SET = {
    "bottomknob": BOTTOMKNOB,
    "topknob": TOPKNOB,
    "lightswitch": LIGHTSWITCH,
    "hinge": HINGE,
    "slide": SLIDE,
    "microwave": MICROWAVE,
    "kettle": KETTLE,
}