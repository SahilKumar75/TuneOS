"""TuneOS — Theme preference state."""

import reflex as rx


class ThemeState(rx.State):
    """Persists the user's theme preference and applies it post-hydration."""

    theme_preference: str = rx.LocalStorage("system", name="tune_theme_pref")

    @rx.event
    def init_theme(self):
        """Called on page on_load (post-hydration). Syncs stored preference."""
        if self.theme_preference == "system":
            yield rx.call_script(
                "window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'",
                callback=ThemeState.apply_resolved_mode,
            )
        # explicit light/dark: head script already applied correct classes from
        # localStorage['theme']; nothing to do until user changes preference.

    @rx.event
    def apply_resolved_mode(self, mode: str):
        """Callback from call_script — receives 'dark' or 'light'.
        Only reloads when localStorage actually needs to change to avoid a loop."""
        safe = "dark" if mode == "dark" else "light"
        yield rx.call_script(
            f"""
            var current = localStorage.getItem('theme');
            if (current !== '{safe}') {{
                localStorage.setItem('theme', '{safe}');
                window.location.reload();
            }}
            """
        )

    @rx.event
    def set_theme(self, pref: str):
        """Called from the settings panel. Updates preference and applies immediately."""
        self.theme_preference = pref
        if pref == "system":
            yield rx.call_script(
                "window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'",
                callback=ThemeState.apply_resolved_mode,
            )
        else:
            safe = "dark" if pref == "dark" else "light"
            yield rx.call_script(
                f"localStorage.setItem('theme', '{safe}'); window.location.reload();"
            )
