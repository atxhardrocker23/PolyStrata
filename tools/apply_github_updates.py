import re
import urllib.request
import json
from pathlib import Path

def apply_web_updates():
    bases = [Path(r"G:\Strata-3060\am2strata\web"), Path(r"G:\Strata-3060\serve\web")]

    for base in bases:
        if not base.exists():
            continue

        # 1. app.css
        css_file = base / "app.css"
        if css_file.exists():
            css = css_file.read_text(encoding="utf-8")
            if ".view--requests" not in css:
                css = css.replace(
                    ".view--monitor, .view--about { overflow-y: auto; }",
                    ".view--monitor, .view--about, .view--requests { overflow-y: auto; }\n.requests-frame { flex: 1; width: 100%; border: 0; }"
                )
            if ".st-header .st-tabs { min-width: 0;" not in css:
                css = css.replace(
                    "/* ---------------------------------------------------------------- narrow screens */",
                    "/* ---------------------------------------------------------------- narrow screens */\n/* tabs that do not fit the header (a narrow phone, or Images on) scroll instead of widening the page */\n.st-header .st-tabs { min-width: 0; overflow-x: auto; scrollbar-width: none; }"
                )
            css_file.write_text(css, encoding="utf-8")
            print(f"[OK] Patched {css_file}")

        # 2. monitor.html
        mhtml_file = base / "monitor.html"
        if mhtml_file.exists():
            mhtml = mhtml_file.read_text(encoding="utf-8")
            if ".embedded header{display:none}" not in mhtml:
                mhtml = mhtml.replace(
                    "header{height:70px;padding:0 32px;",
                    ".embedded header{display:none}header{height:70px;padding:0 32px;"
                )
                mhtml_file.write_text(mhtml, encoding="utf-8")
                print(f"[OK] Patched {mhtml_file}")

        # 3. monitor.js
        mjs_file = base / "monitor.js"
        if mjs_file.exists():
            mjs = mjs_file.read_text(encoding="utf-8")
            if "const sync = () =>" not in mjs:
                old_target = 'let apiKey = sessionStorage.getItem("strata.monitor.key") || "";'
                new_replacement = (
                    '// #142: inside the web app\'s Requests tab, the app\'s header, theme and saved API key stand in for this page\'s own\n'
                    'let host = null;\n'
                    'try { if (window.parent !== window) host = parent.document.documentElement; } catch (e) { /* framed by another site */ }\n'
                    'let apiKey = sessionStorage.getItem("strata.monitor.key") || (host && JSON.parse(localStorage.getItem("strata.apikey") || \'""\')) || "";\n'
                    'document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";\n'
                    'if (host) {\n'
                    '  const sync = () => { document.documentElement.dataset.theme = host.dataset.theme; };\n'
                    '  document.documentElement.classList.add("embedded");\n'
                    '  sync();\n'
                    '  new MutationObserver(sync).observe(host, {attributes: true, attributeFilter: ["data-theme"]});\n'
                    '}'
                )
                if old_target in mjs:
                    mjs = mjs.replace(old_target, new_replacement)
                    mjs_file.write_text(mjs, encoding="utf-8")
                    print(f"[OK] Patched {mjs_file}")

        # 4. index.html
        idx_file = base / "index.html"
        if idx_file.exists():
            idx = idx_file.read_text(encoding="utf-8")
            if 'id="tab-btn-requests"' not in idx:
                target_nav = '<button class="st-tab" role="tab" id="tab-btn-about"'
                requests_nav_btn = (
                    '<button class="st-tab" role="tab" id="tab-btn-requests" aria-controls="view-requests" aria-selected="false" data-tab="requests">\n'
                    '      <svg class="st-icon st-icon--sm"><use href="web/sprite.svg#i-layers"/></svg>Requests</button>\n    '
                )
                idx = idx.replace(target_nav, requests_nav_btn + target_nav)

            if 'id="view-requests"' not in idx:
                target_section = '<!-- ============================ About ============================ -->'
                requests_section = (
                    '<!-- ============================ Requests ============================ -->\n'
                    '  <section class="view view--requests" id="view-requests" role="tabpanel" aria-labelledby="tab-btn-requests" hidden>\n'
                    '    <iframe class="requests-frame" id="requests-frame" title="API monitor" hidden></iframe>\n'
                    '    <div class="about" id="requests-off" hidden>\n'
                    '      <div class="st-card">\n'
                    '        <span class="card-title">The API monitor is off</span>\n'
                    '        <p class="muted">It keeps the last 100 requests\' prompts and answers in memory, so it is off by default. To turn it\n'
                    '          on, add <code>"api_monitor": true</code> to the model\'s config file (<code>strata-&lt;model&gt;.json</code>) or\n'
                    '          start the server with <code>--api-monitor</code>, then restart it.</p>\n'
                    '      </div>\n'
                    '    </div>\n'
                    '  </section>\n\n  '
                )
                idx = idx.replace(target_section, requests_section + target_section)

            idx_file.write_text(idx, encoding="utf-8")
            print(f"[OK] Patched {idx_file}")

        # 5. app.js
        app_file = base / "app.js"
        if app_file.exists():
            appjs = app_file.read_text(encoding="utf-8")
            if "showRequests" not in appjs:
                old_func = (
                    'let tab = "chat";\n'
                    'function showTab(name) {\n'
                    '  tab = ["chat", "images", "monitor", "about"].includes(name) ? name : "chat";\n'
                    '  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));\n'
                    '  for (const v of ["chat", "images", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;'
                )
                new_func = (
                    'let tab = "chat";\n'
                    'const TABS = ["chat", "images", "monitor", "requests", "about"];\n'
                    'function showTab(name) {\n'
                    '  tab = TABS.includes(name) ? name : "chat";\n'
                    '  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));\n'
                    '  for (const v of TABS) $(`view-${v}`).hidden = v !== tab;\n'
                    '  if (tab === "images") $("img-prompt").focus();\n'
                    '  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);\n'
                    '  if (tab === "chat") $("input").focus();\n'
                    '  if (tab === "monitor") loadMcp();\n'
                    '  if (tab === "about") loadConfig();\n'
                    '  if (tab === "requests") showRequests();\n'
                    '  if (lastMetrics) render(lastMetrics);\n'
                    '}\n'
                    '// #142: the Requests tab shows the /api-monitor page, loaded the first time the tab opens, or how to turn it on.\n'
                    'function showRequests() {\n'
                    '  const on = (health.api_monitor !== undefined) ? health.api_monitor : true, frame = $("requests-frame");\n'
                    '  $("requests-off").hidden = on !== false;\n'
                    '  frame.hidden = on !== true;\n'
                    '  if (on && !frame.getAttribute("src")) frame.src = "api-monitor";\n'
                    '}'
                )
                if old_func in appjs:
                    appjs = appjs.replace(old_func, new_func)
                else:
                    # Generic regex fallback for TABS
                    appjs = re.sub(
                        r'tab\s*=\s*\["chat",\s*"images",\s*"monitor",\s*"about"\]\.includes\(name\)\s*\?\s*name\s*:\s*"chat";',
                        'const TABS = ["chat", "images", "monitor", "requests", "about"]; tab = TABS.includes(name) ? name : "chat";',
                        appjs
                    )
                    appjs = appjs.replace('for (const v of ["chat", "images", "monitor", "about"])', 'for (const v of TABS)')
                    if "if (tab === \"requests\") showRequests();" not in appjs:
                        appjs = appjs.replace(
                            'if (tab === "about") loadConfig();',
                            'if (tab === "about") loadConfig();\n  if (tab === "requests") showRequests();'
                        )
                        requests_fn = (
                            '\nfunction showRequests() {\n'
                            '  const on = (health.api_monitor !== undefined) ? health.api_monitor : true, frame = $("requests-frame");\n'
                            '  $("requests-off").hidden = on !== false;\n'
                            '  frame.hidden = on !== true;\n'
                            '  if (on && !frame.getAttribute("src")) frame.src = "api-monitor";\n'
                            '}\n'
                        )
                        appjs = appjs.replace('for (const b of document.querySelectorAll(".st-tab")) b.onclick', requests_fn + 'for (const b of document.querySelectorAll(".st-tab")) b.onclick')

                if 'if (tab === "requests") showRequests();' not in appjs:
                    appjs = appjs.replace(
                        'if (location.hash === "#images" && health.image_generation) showTab("images");',
                        'if (location.hash === "#images" && health.image_generation) showTab("images");\n    if (tab === "requests") showRequests();'
                    )

                app_file.write_text(appjs, encoding="utf-8")
                print(f"[OK] Patched {app_file}")

def apply_poly_server_updates():
    """Apply PR 138 (middle truncation), e4e62b19 (Gemma slot release), and a82999fd (health api_monitor)."""
    server_file = Path(r"G:\Strata-3060\serve\server.py")
    runconfig_file = Path(r"G:\Strata-3060\serve\runconfig.py")

    if runconfig_file.exists():
        rc = runconfig_file.read_text(encoding="utf-8")
        if "truncate_middle" not in rc:
            rc = rc.replace(
                '("drop_oldest", "bool", "Leave out a conversation\'s oldest turns when it no longer fits the context, instead of "\n                            "answering 400 (the system prompt and the last turn stay)"),',
                '("drop_oldest", "bool", "Leave out a conversation\'s oldest turns when it no longer fits the context, instead of "\n                            "answering 400 (the system prompt and the last turn stay)"),\n    ("truncate_middle", "bool", "Trim the middle of an oversized text prompt, keeping its beginning and ending"),'
            )
            runconfig_file.write_text(rc, encoding="utf-8")
            print(f"[OK] Patched {runconfig_file}")

    if server_file.exists():
        srv = server_file.read_text(encoding="utf-8")
        # 1. api_monitor in /health
        if '"image_generation": svc.images is not None, "api_monitor": svc.api_monitor,' not in srv:
            srv = srv.replace(
                '"image_generation": svc.images is not None,',
                '"image_generation": svc.images is not None, "api_monitor": svc.api_monitor,'
            )

        # 2. truncate_middle in Service init
        if "self.truncate_middle = False" not in srv:
            srv = srv.replace(
                "self.drop_oldest = False                      # #17 (opt-in): a conversation too long loses its oldest turns",
                "self.drop_oldest = False                      # #17 (opt-in): a conversation too long loses its oldest turns\n        self.truncate_middle = False                  # opt-in: keep a prompt's beginning and ending when it will not fit"
            )

        # 3. truncate_prompt_middle method
        if "def truncate_prompt_middle(self, ids, max_new):" not in srv:
            target = "def drop_oldest_turns(self, messages, tools, kwargs, ids, max_new):"
            method_code = '''    def truncate_prompt_middle(self, ids, max_new):
        """With "truncate_middle": true, keep the prompt's beginning and ending when text alone exceeds context."""
        ctx = self.engine.max_context or getattr(self.engine, "known_ctx", 0)
        if ctx <= 0:
            return ids
        answer_room = max_new if max_new is not None and max_new > 0 else ctx // 8
        prompt_room = ctx - CTX_SLACK - answer_room
        if prompt_room < 1 or len(ids) <= prompt_room:
            return ids
        keep_start = (prompt_room + 1) // 2
        keep_end = prompt_room - keep_start
        removed = len(ids) - prompt_room
        print(f"[strata] the prompt did not fit the context ({ctx} tokens): its {removed} middle tokens were "
              f"left out (truncate_middle)", flush=True)
        return ids[:keep_start] + (ids[-keep_end:] if keep_end else [])

'''
            srv = srv.replace(target, method_code + target)

        # 4. call truncate_prompt_middle in prepare
        if "if self.truncate_middle and not images:" not in srv:
            srv = srv.replace(
                'images = images_of(messages)\n        if images:',
                'images = images_of(messages)\n        if self.truncate_middle and not images:\n            ids = self.truncate_prompt_middle(ids, max_new)\n        if images:'
            )

        # 5. eos_ids and slot hand-over improvements
        if "eos_ids = frozenset(EOS_IDS)" not in srv:
            srv = srv.replace(
                "turn_ids = frozenset()               # Service: the tokens that open a chat turn (turn_key()); survives restart()",
                "turn_ids = frozenset()               # Service: the tokens that open a chat turn (turn_key()); survives restart()\n    eos_ids = frozenset(EOS_IDS)"
            )

        server_file.write_text(srv, encoding="utf-8")
        print(f"[OK] Patched {server_file}")

if __name__ == "__main__":
    print("--- Applying Web Updates ---")
    apply_web_updates()
    print("--- Applying Server Updates ---")
    apply_poly_server_updates()
    print("--- Done ---")
