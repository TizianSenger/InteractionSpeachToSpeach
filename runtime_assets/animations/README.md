# Animations Folder Structure

Each animation has its own folder containing:
- `animation.vrma` (required) - The VRMA animation file
- `sound.mp3`, `sound.wav`, or `sound.ogg` (optional) - Audio file that plays with the animation

## Current Animations

1. **Angry** - `animations/Angry/`
2. **Blush** - `animations/Blush/`
3. **Clapping** - `animations/Clapping/`
4. **Dance1** - `animations/Dance1/`
5. **Dance2** - `animations/Dance2/`
6. **Dance3** - `animations/Dance3/`
7. **Goodbye** - `animations/Goodbye/`
8. **Jump** - `animations/Jump/`
9. **LookAround** - `animations/LookAround/`
10. **Relax** - `animations/Relax/`
11. **Run** - `animations/Run/` (loops 5 times)
12. **Sad** - `animations/Sad/`
13. **Sleepy** - `animations/Sleepy/`
14. **SlowRun** - `animations/SlowRun/` (loops 5 times)
15. **Surprised** - `animations/Surprised/`
16. **Thinking** - `animations/Thinking/`
17. **Walk** - `animations/Walk/` (loops 5 times, used as idle animation at 0.5x speed)

## Adding New Animations

1. Create a new folder in `web/animations/` (e.g., `MyAnimation/`)
2. Add `animation.vrma` to the folder
3. Optionally add `sound.mp3` (or .wav/.ogg) for synchronized audio
4. The animation will be automatically detected and added to the UI

## Audio File Priority

If multiple audio files exist, the priority is:
1. `sound.mp3`
2. `sound.wav`
3. `sound.ogg`

## Migration from Old Structure

Old structure: `web/VRMA/*.vrma`
New structure: `web/animations/[Name]/animation.vrma`

To migrate:
1. Create folder: `web/animations/[AnimationName]/`
2. Move `[AnimationName].vrma` → `web/animations/[AnimationName]/animation.vrma`
3. Add optional `sound.mp3` if desired
