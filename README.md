# NoteForge

## Goal

1. Take orchestral music input and seperate into different parts using blind source separation (violins, viola, cello, bass)
2. Convert seperate parts into note number and durations
3. Take note number and durations and covert to sheet music

## Workspace Outline

### main.py
- For integration functions ONLY
- Should be able to run main and achive any one of the 3 goals independently and 
- CLI

### bss.py
- Inputs: Mono waveform
- Output: Different parts (<= 4 different audio paths)
- Actual audio files can be stored in ./{music}/seperated_audio_files

### note_conversion.py
- Inputs: Different audio paths
- Outputs (2): Note numbers + durations, sheet music


### File structure for music

```md
output
└── SwanLake
    ├── SwanLake.csv
    ├── SwanLake.xml
    ├── SwanLake_mono.wav
    └── seperated
        ├── SwanLake_1.wav
        ├── SwanLake_2.wav
        ├── SwanLake_3.wav
        └── SwanLake_4.wav
└── SwampThang
    ├── SwampThang.csv
    ├── SwampThang.xml
    ├── SwampThang_mono.wav
    └── seperated
        ├── SwampThang_1.wav
        ├── SwampThang_2.wav
        ├── SwampThang_3.wav
        └── SwampThang_4.wav
```
