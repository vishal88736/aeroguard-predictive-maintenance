# AeroGuard visual assets

The current blue/orange light theme follows the user's AeroGuard design reference. The previous bridge theme remains only in the retained legacy `web/` client.

- `frontend/assets/turbofan-hero.png`: generated using the built-in image generation tool for this project, then copied from its default generated-images location. It illustrates an unbranded industrial engine; it is not a photograph of a monitored engine or evidence of a physical sensor feed.
- `frontend/assets/aeroguard.svg`: code-native turbine brand mark.
- `frontend/vendor/InterVariable.woff2`: Inter variable font from the official Inter repository; SIL Open Font License in `INTER_LICENSE.txt`.
- `frontend/vendor/icons.js`: selected published Lucide static SVG icons; ISC license in `LUCIDE_LICENSE.txt`. Assets load locally without runtime CDN requests.

Final prompt, built-in tool mode:

> Photorealistic premium industrial product photograph for an aerospace predictive-maintenance website hero. A modern unbranded high-bypass turbofan jet engine on a maintenance stand inside a spotless bright aircraft hangar, detailed metallic fan blades and visible precision engineering, subtle warm orange highlights along the casing, cool pale blue daylight. Wide landscape 16:9 composition. Engine occupies the right two thirds with front three-quarter view; the left third is a clean softly lit pale blue-white hangar wall with negative space for real HTML text. High fidelity realistic metal, restrained contrast, editorial engineering photography, professional enterprise software aesthetic. No people, no lettering, no text, no interface, no logos, no diagram, no charts, no watermarks. Asset only, not a website screenshot.

## Public repository scope

The original university reference image and supplied papers are excluded. The retained legacy client uses the authored `web/assets/bridge-illustration.svg`; current screenshots in `docs/images/` show the locally recorded AeroGuard interface. They contain no credentials or supplied reference artwork. Dataset/model artifacts are not distributed with those previews.
