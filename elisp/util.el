;; utilities -- intended to be loaded on .emacs

(defun ce/org-execute-named-block (name)
  (save-excursion
    (org-babel-goto-named-src-block name)
    (org-babel-execute-src-block)
    )
  )


(defun ce/org-table-kill-field ()
  "Kill the current table field or active region. Based on org-table-blank-field."
  (interactive)
  (org-table-check-inside-data-field)
  (if (and (called-interactively-p 'any) (org-region-active-p))
      (let (org-table-clip)
	(org-table-cut-region (region-beginning) (region-end)))
    (skip-chars-backward "^|")
    (backward-char 1)
    (if (looking-at "|[^|\n]+")
	(let* ((pos (match-beginning 0))
	       (match (match-string 0))
	       (len (org-string-width match)))
          (kill-region (+ 1 pos) (+ len pos))
          (org-table-align)
	  ))))

(defun ce/org-capture-current-file ()
  "Add current file as a target to `org-capture'."
  (interactive)
  (let
      ((org-capture-templates
        (append
         org-capture-templates
         '(("c" "Add to current file")
           ("ct" "Todo" entry (file+olp (buffer-file-name) "TODOList") "* TODO %? ")
           ("cl" "Log" entry (file+datetree (buffer-file-name) "Log") "*  %? ")))))
    (org-capture)))

(defun ce/copy-org-code-block ()
  "Copy an org-code-block."
  ;; based on er/mark-org-code-block
  (interactive)
  (let ((case-fold-search t)
        (re "#\\+begin_\\(\\sw+\\)"))
    (unless (looking-at re)
      (search-backward-regexp re))
    (forward-line)
    (set-mark (point))
    (search-forward (concat "#+end_" (match-string 1)))
    (forward-line 0)
    (exchange-point-and-mark)
    (copy-region-as-kill (mark) (point))))

;; attempted pull request of this thing to org-core.el; see github for more info
(defun ce/org-babel-execute-src-block-region (beg end)
  "Execute region in the current source code block.
`org-babel-execure-src-block' is called; the only change is that
only the active region is sent, instead of the whole block."
  (interactive "r")
  (if (ce/org-babel-is-region-within-src-block beg end)
      (let ((info (org-babel-get-src-block-info)))
        (setcar (nthcdr 1 info) (buffer-substring beg end))
        (org-babel-execute-src-block nil info))
    (message "Region not in src-block!")))

(defun ce/org-babel-is-region-within-src-block (beg end)
  "Check if region is within a single src-block.
Block header and footer are ignored, so we are checking for the
source code only.
Used by `ce/org-babel-execute-src-block-region' to check if region
is executable."
  (save-excursion
    (eq
     (progn
       (goto-char beg)
       (forward-line -1)
       (org-babel-where-is-src-block-head))
     (progn
       (goto-char end)
       (forward-line 1)
       (org-babel-where-is-src-block-head)))))

(defun ce/open-corresponding-pdf ()
  "Open pdf corresponding to current file in external app."
  (interactive)
  (let ((ext (file-name-extension (buffer-file-name)))
        (base (concat
               (file-name-directory (buffer-file-name))
               (file-name-base (buffer-file-name)))))
    (start-process "" nil "xdg-open" (concat base ".pdf"))))

(defun ce/switch-to-pdf-or-tex ()
  "Switch buffers; from tex to pdf or from pdf to tex."
  (interactive)
  (let ((ext (file-name-extension (buffer-file-name)))
        (base (concat
               (file-name-directory (buffer-file-name))
               (file-name-base (buffer-file-name)))))
    (cond
     ((string= ext "tex") (find-file (concat base ".pdf")))
     ((string= ext "pdf") (find-file (concat base ".tex")))
     (t (message "Not in tex or pdf file.")))))

;; TODO: refactor! same as switch-to-pdf-org-tex!
(defun ce/switch-to-pdf-or-org ()
  "Switch buffers; from org to pdf or from pdf to org."
  (interactive)
  (let ((ext (file-name-extension (buffer-file-name)))
        (base (concat
               (file-name-directory (buffer-file-name))
               (file-name-base (buffer-file-name)))))
    (cond
     ((string= ext "org") (find-file (concat base ".pdf")))
     ((string= ext "pdf") (find-file (concat base ".org")))
     (t (message "Not in org or pdf file.")))))

(defun ce/other-python-shell ()
  "Rename current python shell and start another one. "
  (interactive)
  (switch-to-buffer "*Python*")
  (rename-uniquely)
  (run-python (python-shell-parse-command))
  (switch-to-buffer "*Python*"))

(defun ce/restart-python-shell (arg)
  "Kill current python shell and start another one. "
  (interactive "P")
  (kill-buffer "*Python*")
  (if (eq arg nil)
      (run-python "ipython3")
    (run-python "ipython"))
  (switch-to-buffer "*Python*"))

(defun ce/python-close-all ()
  "Close all pyplot windows by calling plt.close('all')"
  (interactive)
  (python-shell-send-string "plt.close('all')"))

(defun ce/matlab-shell-close-all ()
  "Close all matlab plots. This command requires an active MATLAB shell."
  (interactive)
  (matlab-shell-collect-command-output "close all;"))

(defun ce/matlab-shell-describe-command ()
  "Describe function or script at point through matlab-shell-describe-command."
  (interactive)
  (matlab-shell-describe-command
   (symbol-name (symbol-at-point))))

(defun ce/matlab-shell-edit-file ()
  "Edit function or script at point through matlab-shell edit command."
  (interactive)
  (matlab-shell-run-command
   (concat "edit " (symbol-name (symbol-at-point)))))

(defun ce/pdf-tools-org-export-to-org-directory (directory)
  "Calls `pdf-tools-org-export-to-org` on all pdfs of a directory."
  (mapc (lambda (filename)
          (find-file filename)
          (pdf-tools-org-export-to-org)
          (kill-current-buffer)
          )
        (directory-files directory t "pdf$")))

;; grep on all bibtex-related org files
(defun ce/grep-bibtex-org (regexp)
  "Grep for regexp on all org-files in `org-ref-pdf-directory'"
  (interactive "s")
  (grep-compute-defaults) ;; lgrep fails without this
  (lgrep regexp "*.org" (expand-file-name org-ref-pdf-directory)))

;; search region on google scholar
;; will someday evolve to a org-ref-scholar-to-bibtex-pdf
(defun ce/search-region-google-scholar (beg end)
  (interactive "r")
  (browse-url
   (concat
    "https://scholar.google.com/scholar?q="
    (replace-regexp-in-string
     " "
     "+"
     (if (eq major-mode 'pdf-view-mode)
         ;; pdf-view-active-region-text returns list of hopefully one string
         (car (pdf-view-active-region-text))
       (buffer-substring-no-properties beg end))))))

(defun ce/www-get-page-title (url)
  "Return <title> from url. "
  (let ((title))
    (with-current-buffer (url-retrieve-synchronously url)
      (goto-char (point-min))
      (re-search-forward "<title>\\([^<]*\\)</title>" nil t 1)
      (setq title (match-string 1))
      (goto-char (point-min))
      (re-search-forward "charset='?\\([-0-9a-zA-Z]*\\)'?" nil t 1)
      (decode-coding-string title (intern (downcase (match-string 1)))))))

(defun ce/org-link-change-title ()
  "Replace next org-mode link caption with the title of the page. "
  (interactive)
  (let ((regex "\\[\\[\\(.*?\\)\\]\\[.*?\\]\\]") url title)
    ;; www-get-page-title will mess with matches; so we must save excursion and search again.
    (save-excursion
      (re-search-forward regex)
      (setq url (match-string 1))
      (setq title
            (replace-regexp-in-string "\\s-+" " " (ce/www-get-page-title url))))
    (re-search-forward regex)
    (replace-match (concat "[["
                           url
                           "]["
                           title
                           "]]"))))
;;"[[\1][\,(www-get-page-title \1)]]"

;; from: http://stackoverflow.com/questions/12165205/how-to-copy-paste-a-region-from-emacs-buffer-with-line-file-reference
(defun ce/kill-with-linenum (beg end)
  (interactive "r")
  (save-excursion
    (goto-char end)
    (skip-chars-backward "\n \t")
    (setq end (point))
    (let* ((chunk (buffer-substring beg end))
           (chunk (concat
                   (format "+-------- line: %-d - %s --\n| "
                           (line-number-at-pos beg)
                           (or (buffer-file-name) (buffer-name))
                           )
                   (replace-regexp-in-string "\n" "\n| " chunk)
                   (format "\n+-------- line: %-d --"
                           (line-number-at-pos end)))))
      (kill-new chunk)))
  (deactivate-mark))

;; from: http://whattheemacsd.com/file-defuns.el-02.html
(defun ce/delete-current-buffer-file ()
  "Removes file connected to current buffer and kills buffer."
  (interactive)
  (let ((filename (buffer-file-name))
        (buffer (current-buffer))
        (name (buffer-name)))
    (if (not (and filename (file-exists-p filename)))
        (ido-kill-buffer)
      (when (yes-or-no-p "Are you sure you want to remove this file? ")
        (delete-file filename)
        (kill-buffer buffer)
        (message "File '%s' successfully removed" filename)))))

;; from: http://whattheemacsd.com//file-defuns.el-01.html
(defun ce/rename-current-buffer-file ()
  "Renames current buffer and file it is visiting."
  (interactive)
  (let ((name (buffer-name))
        (filename (buffer-file-name)))
    (if (not (and filename (file-exists-p filename)))
        (error "Buffer '%s' is not visiting a file!" name)
      (let ((new-name (read-file-name "New name: " filename)))
        (if (get-buffer new-name)
            (error "A buffer named '%s' already exists!" new-name)
          (rename-file filename new-name 1)
          (rename-buffer new-name)
          (set-visited-file-name new-name)
          (set-buffer-modified-p nil)
          (message "File '%s' successfully renamed to '%s'"
                   name (file-name-nondirectory new-name)))))))

(defun ce/copy-directory-to-kill-ring()
  "copy current file dir (add to kill ring)"
  (interactive)
  (when buffer-file-name
    (kill-new 
     (file-name-directory (buffer-file-name)))))

(defun ce/insert-date()
  "Inserts output of date command in buffer"
  (interactive)
  (insert (format-time-string "%Y-%m-%d %H:%M:%S")))

;; from: http://stackoverflow.com/questions/18121808/emacs-ediff-marked-files-in-different-dired-buffers
(defun ce/dired-ediff-marked-files ()
  "Run ediff-files on a pair of files marked in dired buffer"
  (interactive)
  (let* ((marked-files (dired-get-marked-files nil nil))
         (other-win (get-window-with-predicate
                     (lambda (window)
                       (with-current-buffer (window-buffer window)
                         (and (not (eq window (selected-window)))
                              (eq major-mode 'dired-mode))))))
         (other-marked-files (and other-win
                                  (with-current-buffer (window-buffer other-win)
                                    (dired-get-marked-files nil)))))
    (cond ((= (length marked-files) 2)
           (ediff-files (nth 0 marked-files)
                        (nth 1 marked-files)))
          ((and (= (length marked-files) 1)
                (= (length other-marked-files) 1))
           (ediff-files (nth 0 marked-files)
                        (nth 0 other-marked-files)))
          (t (error "mark exactly 2 files, at least 1 locally")))))

(defun ce/toggle-fullscreen (&optional f)
  (interactive)
  (let ((current-value (frame-parameter nil 'fullscreen)))
    (set-frame-parameter nil 'fullscreen
			 (if (equal 'fullboth current-value)
			     (if (boundp 'old-fullscreen) old-fullscreen nil)
			   (progn (setq old-fullscreen current-value)
				  'fullboth)))))

(defun ce/copy-column-as-list ()
  "Copy the current Org-table column as a Python-style list to the clipboard."
  (interactive)
  (unless (org-at-table-p) (user-error "Not in an org table"))
  (let* ((col-idx (1- (org-table-current-column))) ;; 0-indexed
         (table (org-table-to-lisp))                 ;; Parse table
         ;; Extract column, ignoring hlines ('nil' in lisp structure)
         (col-vals (delq nil
                         (mapcar (lambda (row)
                                   (when (listp row) (nth col-idx row)))
                                 table)))
         ;; Format as [1, 2, 3]
         (result (format "[%s]" (mapconcat #'identity col-vals ", "))))

    (kill-new result)
    (message "Copied column: %s" result)))

(defun ce/attach-latest-pdf-to-bib-entry ()
  "Move the newest PDF from Downloads to the library, named after the latest BibTeX entry."
  (interactive)
  (let* ((bib-file ce/helm-bibtex-bibliography)
         (library-path (car ce/helm-bibtex-library-path))
         (downloads-dir "~/Downloads/")
         ;; 1. Find the most recent citation key in the bib file
         (latest-key
          (with-temp-buffer
            (insert-file-contents bib-file)
            (goto-char (point-max))
            ;; Regex to find the last entry key: @type{KEY,
            (if (re-search-backward "^@[a-zA-Z]+{\\([^,]+\\)," nil t)
                (match-string 1)
              (error "Could not find a valid BibTeX entry in %s" bib-file))))
         ;; 2. Find the newest PDF in Downloads
         (newest-pdf
          (let ((files (directory-files-and-attributes downloads-dir t "\\.pdf$")))
            ;; Sort by modification time (descending)
            (if files
                (caar (sort files (lambda (a b) (time-less-p (nth 6 b) (nth 6 a)))))
              nil)))
         ;; 3. Prompt for Source File (Default: newest PDF)
         (source-file
          (read-file-name "Source PDF: "
                          downloads-dir
                          newest-pdf
                          t
                          (file-name-nondirectory newest-pdf)))
         ;; 4. Prompt for Target Name (Default: latest-key)
         (target-name
          (read-string (format "Target filename (without .pdf): ")
                       latest-key))
         ;; Construct full target path
         (target-path (expand-file-name (concat target-name ".pdf") library-path)))
    ;; 5. Execute Move
    (when (file-exists-p target-path)
      (if (y-or-n-p (format "File %s already exists. Overwrite? " target-path))
          (delete-file target-path)
        (user-error "Aborted.")))

    (rename-file source-file target-path)
    (message "Moved '%s' to '%s'" (file-name-nondirectory source-file) target-path)))

(defun ce/prompt-vterm-buffer ()
  "Prompt the user to select a vterm buffer, defaulting to the most recent."
  (let ((vterms (seq-filter (lambda (name) (string-match-p "vterm" name))
                            (mapcar #'buffer-name (buffer-list)))))
    (unless vterms (error "No vterm buffers found!"))
    ;; Defaults to the first item (most recently active vterm buffer)
    (completing-read "Target vterm: " vterms nil t nil nil (car vterms))))

(defun ce/gemini-cli-send-string (text target-buffer &optional auto-return)
  "Send TEXT to TARGET-BUFFER using bracketed paste."
  (unless (get-buffer target-buffer)
    (error "Buffer %s not found." target-buffer))
  (with-current-buffer target-buffer
    ;; Bracketed paste prevents the CLI from executing prematurely on newlines
    (kill-new text)
    (vterm-yank)
    (when auto-return (vterm-send-return)))
  (pop-to-buffer target-buffer))

(defun ce/gemini-cli-send-region (beg end target-buffer prompt-text)
  "Send PROMPT-TEXT and optional buffer context to a gemini-cli vterm buffer.
If a region is active, sends the region text and line numbers.
If no region is active, in an org-mode file, and on a subtree heading, sends
only the subtree content.
If no region is active but the buffer visits a file, sends the file name.
If no region is active and no file is visited, sends only the prompt.
File paths are made relative to the target vterm's current directory."
  (interactive
   (let* ((has-region (use-region-p))
          ;; Detect if we are on an org heading with no active region
          (is-org-heading (and (not has-region)
                               (derived-mode-p 'org-mode)
                               (org-at-heading-p))))
     (list (when has-region (region-beginning))
           (when has-region (region-end))
           (ce/prompt-vterm-buffer)
           ;; Skip prompting for text if we just intend to send the org subtree
           (if is-org-heading "" (read-string "Prompt: ")))))

  (unless (get-buffer target-buffer)
    (error "Buffer %s not found." target-buffer))

  (let* ((abs-file (buffer-file-name))
         ;; Retrieve the working directory of the target vterm
         (vterm-dir (with-current-buffer target-buffer default-directory))
         ;; Convert absolute path to relative path
         (file (when abs-file (file-relative-name abs-file vterm-dir)))
         (is-org-heading (and (not beg)
                              (not end)
                              (derived-mode-p 'org-mode)
                              (org-at-heading-p)))
         (context
          (cond
           ;; Case 0: Org heading (ignore standard context, handled in payload)
           (is-org-heading nil)
           ;; Case 1: Region is active (send region text)
           ((and beg end)
            (format "File: %s (Lines %d-%d)\n=\n%s\n=\n"
                    (or file (buffer-name))
                    (line-number-at-pos beg)
                    (line-number-at-pos end)
                    (buffer-substring-no-properties beg end)))
           ;; Case 2: No region, but associated to a file (send file name)
           (file
            (format "File: %s\n" file))
           ;; Case 3: No region, not a file
           (t nil)))

         ;; Assemble the final string to send
         (payload
          (cond
           ;; Case 0: On an Org heading, we ONLY send the subtree content
           (is-org-heading
            (save-excursion
              (save-restriction
                (org-narrow-to-subtree)
                (buffer-substring-no-properties (point-min) (point-max)))))
           ((and context (not (string-empty-p prompt-text)))
            (format "%s\n\n%s" prompt-text context))
           (context context)
           (t (format "%s\n" prompt-text)))))

    ;; Avoid sending completely empty prompts if no context/prompt was given
    (unless (or (null payload) (string= payload "") (string= payload "\n"))
      (ce/gemini-cli-send-string payload target-buffer nil))))
