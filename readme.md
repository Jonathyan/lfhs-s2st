# Nederlands → Indonesisch S2ST MVP met SeamlessM4T v2

Deze MVP (Minimal Viable Product) biedt een oplossing voor spraak-naar-spraak vertaling (S2ST) van Nederlands naar Indonesisch, speciaal voor vertalingen van preken. De oplossing maakt gebruik van Meta's SeamlessM4T v2 framework en kan volledig lokaal draaien op een MacBook Pro M1 Pro.

## Belangrijkste kenmerken

- **Speech-to-Speech Translation**: Direct van Nederlandse spraak naar Indonesische spraak
- **Voice Cloning**: Optionele stemkloning voor natuurlijkere output
- **Volledig lokaal**: Draait volledig op je MacBook Pro M1 zonder afhankelijkheid van cloud services
- **Batch verwerking**: Geschikt voor lange opnames (~1,5 uur)
- **Optimalisaties voor M1 Pro**: Gebruikt Metal Performance Shaders (MPS) voor optimale prestaties

## Systeemvereisten

- MacBook Pro met M1 Pro chip
- macOS 12 (Monterey) of nieuwer
- Python 3.9+
- ~8GB vrij geheugen
- ~10GB vrije schijfruimte
- ffmpeg (wordt automatisch geïnstalleerd via setup.sh)

## Installatie

1. Clone of download deze repository
2. Zorg dat Homebrew is geïnstalleerd (https://brew.sh)
3. Voer het setup script uit:

```bash
chmod +x setup.sh
./setup.sh
```

## Gebruik

### Voorbereiding

1. Plaats je Nederlandse preek-opname in de `input` map als `dutch_sermon.wav`
   - Ondersteunde formats: WAV, MP3, FLAC (voorkeur: WAV, 16kHz, mono)
   - Voor optimale resultaten: zorg voor een duidelijke opname met weinig achtergrondgeluid

2. (Optioneel) Voor stemkloning:
   - Plaats een korte (5-10 seconden) opname van de doelstem in de `input` map als `voice_reference.wav`
   - Dit zal worden gebruikt om de vertaalde stem te laten klinken als de referentiestem

### Vertaling starten

Activeer de virtual environment en start de vertaling:

```bash
source venv/bin/activate
python seamless_s2st_mvp.py
```

De voortgang wordt in de terminal weergegeven. Het volledige proces voor een preek van 1,5 uur zal naar verwachting 1-3 uur duren, afhankelijk van de specifieke hardware van je M1 Pro.

### Output

Na voltooiing vind je het vertaalde audiobestand in de `output` map als `indonesian_sermon.wav`.

## Verwachtingen en beperkingen

- **Niet real-time**: Deze MVP is geoptimaliseerd voor kwaliteit, niet voor snelheid
- **Batchverwerking**: Audio wordt verwerkt in segmenten van ~30 seconden
- **Geheugengebruik**: Piekgebruik tot ~6-8GB geheugen tijdens verwerking
- **Kwaliteit**: De vertaalnauwkeurigheid is afhankelijk van:
  - Helderheid van het bronmateriaal
  - Complexiteit van de theologische terminologie
  - Dialectvariaties in het Nederlands
- **Stemkloning**: Kwaliteit van stemkloning hangt af van de lengte en kwaliteit van de referentie-opname

## Tips voor optimale resultaten

- **Audio preprocessing**: Voor betere resultaten kun je de preek eerst voorbewerken met ruisonderdrukking
- **Segmenteren**: Voor zeer lange preken (>2 uur), overweeg om ze op te splitsen in aparte delen
- **Terminologie**: SeamlessM4T v2 heeft beperkingen bij specifieke theologische terminologie, dus hou hier rekening mee
- **Stemkloning**: Gebruik een helder, emotioneel neutraal stemfragment voor de beste kloonresultaten

## Probleemoplossing

- **Out of Memory**: Verlaag `chunk_size_ms` in de code (bijv. van 30000 naar 15000)
- **Vertaalkwaliteit**: Zorg voor betere audiokwaliteit van de bron
- **Installatiefouten**: Zorg dat XCode Command Line Tools is geïnstalleerd

## Toekomstige verbeteringen

Deze MVP kan worden uitgebreid met:
- Interface voor real-time monitoring
- GPU-optimalisaties voor snellere verwerking
- Integratie met spraakpreprocessing voor ruisvermindering
- Fine-tuning voor theologische terminologie
- Ondersteuning voor andere talen

## Credits en licenties

- SeamlessM4T v2 is ontwikkeld door Meta AI Research en gebruikt onder de CC-BY-NC 4.0 licentie
- Deze implementatie is alleen voor niet-commercieel gebruik
