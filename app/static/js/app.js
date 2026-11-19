/* HarvestGuard - small progressive enhancements (no inline JS because of the CSP) */
(function () {
  // real-time WebGL 3D crop-field background (falls back to a plain gradient when WebGL is unavailable)
  var bg = document.getElementById('bg');
  if (bg && window.HGBackground) {
    try { window.HGBackground.start(bg); } catch (e) { document.documentElement.classList.add('no-webgl'); }
  } else { document.documentElement.classList.add('no-webgl'); }
  // show / hide password
  document.querySelectorAll('.pw-toggle').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var input = btn.parentElement.querySelector('input');
      var show = input.type === 'password';
      input.type = show ? 'text' : 'password';
      btn.textContent = show ? 'Hide' : 'Show';
      btn.setAttribute('aria-pressed', String(show));
    });
  });

  // live password-policy checklist (mirrors app/security.py)
  var pw = document.querySelector('[data-pw-check]');
  if (pw) {
    var bar = document.querySelector('.strength');
    var rules = {
      len: function (v) { return v.length >= 8; },
      letter: function (v) { return /[A-Za-z]/.test(v); },
      num: function (v) { return /\d/.test(v); },
      mix: function (v) { return /[A-Z]/.test(v) && /[a-z]/.test(v) || /[^A-Za-z0-9]/.test(v); }
    };
    var update = function () {
      var v = pw.value, score = 0;
      Object.keys(rules).forEach(function (k) {
        var ok = rules[k](v);
        var li = document.querySelector('[data-rule="' + k + '"]');
        if (li) li.classList.toggle('ok', ok);
        if (ok) score++;
      });
      if (v.length >= 12) score = Math.min(4, score + 1);
      if (bar) { bar.dataset.level = v ? String(Math.max(1, Math.min(4, score))) : '0'; bar.firstElementChild.style.width = (v ? Math.max(1, Math.min(4, score)) * 25 : 0) + '%'; }
    };
    pw.addEventListener('input', update); update();
    var conf = document.querySelector('[data-pw-confirm]');
    if (conf) {
      var chk = function () { conf.setCustomValidity(conf.value && conf.value !== pw.value ? 'Passwords do not match' : ''); };
      conf.addEventListener('input', chk); pw.addEventListener('input', chk);
    }
  }

  // leaf upload: drag & drop, preview, busy state
  var drop = document.querySelector('.drop');
  if (drop) {
    var input = drop.querySelector('input[type=file]');
    var prev = document.querySelector('.preview');
    var label = drop.querySelector('[data-file-label]');
    var show = function (file) {
      if (!file) return;
      if (label) label.textContent = file.name + ' (' + Math.round(file.size / 1024) + ' KB)';
      var url = URL.createObjectURL(file);
      prev.src = url; prev.style.display = 'block';
      document.querySelector('[data-submit]').disabled = false;
    };
    input.addEventListener('change', function () { show(input.files[0]); });
    ['dragenter', 'dragover'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add('over'); }); });
    ['dragleave', 'drop'].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove('over'); }); });
    drop.addEventListener('drop', function (ev) {
      if (ev.dataTransfer.files.length) { input.files = ev.dataTransfer.files; show(input.files[0]); }
    });
    var form = drop.closest('form');
    form.addEventListener('submit', function () { form.classList.add('busy'); form.querySelector('[data-submit]').disabled = true; });
  }

  // confirm destructive actions
  document.querySelectorAll('form[data-confirm]').forEach(function (f) {
    f.addEventListener('submit', function (e) { if (!window.confirm(f.dataset.confirm)) e.preventDefault(); });
  });
})();
