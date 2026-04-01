/* ================================================================
   APP.JS — NID OCR System
   Shared API client, utilities, and UX helpers
   ================================================================ */

'use strict';

const API_BASE = 'http://localhost:8000';

/* ----------------------------------------------------------------
   NIDApi — REST client with timeout + unified error handling
   ---------------------------------------------------------------- */
class NIDApi {
  constructor(baseUrl = API_BASE) {
    this.baseUrl    = baseUrl.replace(/\/$/, '');
    this.timeoutMs  = 30_000;
  }

  /* Low-level fetch with AbortController timeout */
  async _fetch(method, path, options = {}) {
    const ctrl  = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);

    try {
      const res = await fetch(this.baseUrl + path, {
        method,
        signal: ctrl.signal,
        ...options,
      });
      clearTimeout(timer);
      return res;
    } catch (err) {
      clearTimeout(timer);
      if (err.name === 'AbortError') {
        throw Object.assign(new Error('Request timed out. Please try again.'), { type: 'timeout' });
      }
      throw Object.assign(err, { type: 'network' });
    }
  }

  /* JSON helper — throws a rich error on non-2xx */
  async _json(method, path, init = {}) {
    const res = await this._fetch(method, path, {
      headers: { 'Content-Type': 'application/json', ...init.headers },
      ...init,
    });

    let body;
    try { body = await res.json(); } catch { body = {}; }

    if (!res.ok) {
      const msg = body.error || body.detail || `HTTP ${res.status}`;
      const err = Object.assign(new Error(msg), {
        status: res.status,
        data:   body,
      });
      throw err;
    }
    return body;
  }

  /* POST multipart — returns parsed JSON response */
  async uploadImage(file) {
    const form = new FormData();
    form.append('image', file);

    const res = await this._fetch('POST', '/api/upload/', { body: form });
    let body;
    try { body = await res.json(); } catch { body = {}; }

    if (!res.ok) {
      const err = Object.assign(
        new Error(body.error || `HTTP ${res.status}`),
        { status: res.status, data: body }
      );
      throw err;
    }
    return body;    // { success, message, data, warnings }
  }

  /* GET /api/records/?... */
  async getRecords(params = {}) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== '' && v != null))
    ).toString();
    return this._json('GET', `/api/records/${qs ? '?' + qs : ''}`);
  }

  /* GET /api/records/:id/ */
  async getRecord(id) {
    return this._json('GET', `/api/records/${id}/`);
  }

  /* PUT /api/records/:id/ */
  async updateRecord(id, data) {
    return this._json('PUT', `/api/records/${id}/`, { body: JSON.stringify(data) });
  }

  /* DELETE /api/records/:id/ */
  async deleteRecord(id) {
    return this._json('DELETE', `/api/records/${id}/`);
  }

  /* GET /api/stats/ */
  async getStats() {
    return this._json('GET', '/api/stats/');
  }

  /* GET /api/export/ — returns Blob */
  async exportRecords(params = {}) {
    const qs = new URLSearchParams(
      Object.fromEntries(Object.entries(params).filter(([, v]) => v !== '' && v != null))
    ).toString();
    const res = await this._fetch('GET', `/api/export/${qs ? '?' + qs : ''}`);
    if (!res.ok) throw new Error(`Export failed: HTTP ${res.status}`);
    return res.blob();
  }
}

/* Singleton instance */
const api = new NIDApi();

/* ----------------------------------------------------------------
   Toast Notifications
   ---------------------------------------------------------------- */
const TOAST_ICONS = {
  success: '✅',
  error:   '❌',
  warning: '⚠️',
  info:    'ℹ️',
};

const TOAST_TITLES = {
  success: 'Success',
  error:   'Error',
  warning: 'Warning',
  info:    'Info',
};

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {number} duration  ms before auto-dismiss (0 = manual only)
 */
function showToast(message, type = 'info', duration = 4000) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] ?? 'ℹ️'}</span>
    <div class="toast-body">
      <div class="toast-title">${TOAST_TITLES[type]}</div>
      <div class="toast-msg">${escHtml(message)}</div>
    </div>
    <button class="toast-close" aria-label="Dismiss">✕</button>
  `;

  const dismiss = () => {
    toast.classList.add('removing');
    toast.addEventListener('animationend', () => toast.remove(), { once: true });
  };

  toast.querySelector('.toast-close').addEventListener('click', dismiss);
  container.appendChild(toast);

  if (duration > 0) setTimeout(dismiss, duration);
  return dismiss;
}

/* ----------------------------------------------------------------
   Modal Helpers
   ---------------------------------------------------------------- */
let _modalBackdrop = null;

/**
 * Show a modal dialog.
 * @param {string} title
 * @param {string} bodyHtml   Raw HTML for the modal body
 * @param {string} footerHtml  Raw HTML for the footer (optional)
 */
function showModal(title, bodyHtml, footerHtml = '') {
  closeModal();

  _modalBackdrop = document.createElement('div');
  _modalBackdrop.className = 'modal-backdrop';
  _modalBackdrop.innerHTML = `
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title-id">
      <div class="modal-header">
        <h2 class="modal-title" id="modal-title-id">${escHtml(title)}</h2>
        <button class="btn-icon-only modal-x-close" aria-label="Close">✕</button>
      </div>
      <div class="modal-body">${bodyHtml}</div>
      ${footerHtml ? `<div class="modal-footer">${footerHtml}</div>` : ''}
    </div>
  `;

  _modalBackdrop.querySelector('.modal-x-close')
    .addEventListener('click', closeModal);

  _modalBackdrop.addEventListener('click', (e) => {
    if (e.target === _modalBackdrop) closeModal();
  });

  document.body.appendChild(_modalBackdrop);
  document.body.style.overflow = 'hidden';

  // Focus trap — focus the first focusable element
  const focusable = _modalBackdrop.querySelector('button, input, select, textarea, [tabindex]');
  if (focusable) setTimeout(() => focusable.focus(), 60);
}

function closeModal() {
  if (_modalBackdrop) {
    _modalBackdrop.remove();
    _modalBackdrop = null;
    document.body.style.overflow = '';
  }
}

// Close on Escape
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeModal();
});

/* ----------------------------------------------------------------
   Formatting Utilities
   ---------------------------------------------------------------- */

/**
 * Format a 0–1 confidence float to "87.0%".
 * @param {number|null} score
 * @returns {string}
 */
function formatConfidence(score) {
  if (score == null) return 'N/A';
  return (score * 100).toFixed(1) + '%';
}

/**
 * Format an ISO datetime string to a human-readable local format.
 * @param {string} iso
 * @returns {string}  e.g. "14 Mar 2026, 10:30 AM"
 */
function formatDate(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-GB', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit',
    }).replace(',', ',');
  } catch {
    return iso;
  }
}

/**
 * Escape HTML special characters.
 * @param {string} str
 * @returns {string}
 */
function escHtml(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Truncate a string to maxLen characters.
 * @param {string} str
 * @param {number} maxLen
 */
function truncate(str, maxLen = 30) {
  if (!str) return '';
  return str.length > maxLen ? str.slice(0, maxLen) + '…' : str;
}

/**
 * Format a file size in bytes to a human-readable string.
 */
function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

/**
 * Return a CSS class name for a confidence score.
 * @param {number} score  0–1
 * @returns {'conf-high'|'conf-mid'|'conf-low'}
 */
function confidenceClass(score) {
  if (score == null) return 'conf-low';
  if (score >= 0.85) return 'conf-high';
  if (score >= 0.60) return 'conf-mid';
  return 'conf-low';
}

/**
 * Return a badge CSS class for a processing status string.
 */
function statusBadgeClass(status) {
  switch (status) {
    case 'SUCCESS':    return 'badge-success';
    case 'FAILED':     return 'badge-error';
    case 'PROCESSING': return 'badge-info';
    default:           return 'badge-neutral';
  }
}

/**
 * Return a CSS class for a blood group string (A+, AB-, etc.).
 */
function bloodGroupClass(bg) {
  if (!bg) return '';
  if (bg.startsWith('AB')) return 'blood-AB';
  if (bg.startsWith('A'))  return 'blood-A';
  if (bg.startsWith('B'))  return 'blood-B';
  return 'blood-O';
}

/* ----------------------------------------------------------------
   Debounce
   ---------------------------------------------------------------- */
/**
 * Returns a debounced version of fn.
 * @param {Function} fn
 * @param {number} delay ms
 */
function debounce(fn, delay = 300) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}

/* ----------------------------------------------------------------
   Generic error message from API error
   ---------------------------------------------------------------- */
function apiErrorMsg(err) {
  if (!navigator.onLine) return 'You appear to be offline.';
  if (err.type === 'timeout') return err.message;
  if (err.type === 'network') return 'Network error. Please check your connection.';
  switch (err.status) {
    case 400: return err.message || 'Invalid request.';
    case 404: return 'Record not found.';
    case 413: return 'File too large.';
    case 415: return err.message || 'Unsupported file format.';
    case 422: return 'Image quality too low for OCR processing.';
    case 500: return 'Server error. Please try again.';
    default:  return err.message || 'An unexpected error occurred.';
  }
}

/* ----------------------------------------------------------------
   Drag-and-drop — prevent browser from opening files
   dropped outside designated zones
   ---------------------------------------------------------------- */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
  document.addEventListener(evt, (e) => {
    if (!e.target.closest('.upload-zone')) {
      e.preventDefault();
      e.stopPropagation();
    }
  });
});

/* ----------------------------------------------------------------
   Online / Offline detection
   ---------------------------------------------------------------- */
function _updateOnlineBanner() {
  const banner = document.getElementById('offline-banner');
  if (!banner) return;
  if (navigator.onLine) {
    banner.classList.remove('visible');
  } else {
    banner.classList.add('visible');
  }
}

window.addEventListener('online',  _updateOnlineBanner);
window.addEventListener('offline', _updateOnlineBanner);

/* Run on load in case we start offline */
document.addEventListener('DOMContentLoaded', _updateOnlineBanner);

/* ----------------------------------------------------------------
   FileReader compatibility check
   ---------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  if (typeof FileReader === 'undefined') {
    showToast(
      'Your browser does not support file uploads. Please use Chrome, Firefox, or Edge.',
      'error',
      0,
    );
  }
});

/* ----------------------------------------------------------------
   Active nav link detection
   ---------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  const page = document.body.dataset.page;
  document.querySelectorAll('.nav-link[data-nav]').forEach(link => {
    if (link.dataset.nav === page) link.classList.add('active');
  });
});

/* ----------------------------------------------------------------
   File download helper (for export blob)
   ---------------------------------------------------------------- */
function downloadBlob(blob, filename) {
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
