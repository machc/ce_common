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

(defun ce/org-export-subtree-to-html5-and-open ()
  "Export the current Org subtree to HTML5 using Pandoc and open it."
  (interactive)
  ;; Arguments: ASYNC SUBTREEP VISIBLE-ONLY BODY-ONLY EXT-PLIST
  ;; We pass `nil` for async, and `t` for subtreep.
  (org-pandoc-export-to-html5-and-open nil t))

(defun ce/org-export-subtree-to-docx ()
  "Export the current Org subtree to docx using Pandoc."
  (interactive)
  (org-pandoc-export-to-docx nil t))

(defun ce/org-export-subtree-to-pptx ()
  "Export the current Org subtree to pptx using Pandoc."
  (interactive)
  (org-pandoc-export-to-pptx nil t))

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

(defun ce/vterm-send-string (text target-buffer &optional auto-return)
  "Send TEXT to TARGET-BUFFER using bracketed paste."
  (unless (get-buffer target-buffer)
    (error "Buffer %s not found." target-buffer))
  (with-current-buffer target-buffer
    ;; Bracketed paste prevents the CLI from executing prematurely on newlines
    (kill-new text)
    (vterm-yank)
    (when auto-return (vterm-send-return)))
  (pop-to-buffer target-buffer))

(defun ce/get-buffer-context (&optional beg end target-dir)
  "Return formatted buffer context (org subtree, region, or filename).
File paths are formatted relative to TARGET-DIR if provided."
  (let* ((abs-file (buffer-file-name))
         (file (when abs-file
                 (if target-dir (file-relative-name abs-file target-dir) abs-file)))
         ;; Guess language from file extension or major mode
         (lang (or (and abs-file (file-name-extension abs-file))
                   (replace-regexp-in-string "-mode$" "" (symbol-name major-mode))))
         (is-org-heading (and (not beg)
                              (not end)
                              (derived-mode-p 'org-mode)
                              (org-at-heading-p))))
    (cond
     ;; Case 0: Org heading (return subtree content)
     (is-org-heading
      (save-excursion
        (save-restriction
          (org-narrow-to-subtree)
          (buffer-substring-no-properties (point-min) (point-max)))))

     ;; Case 1: Region is active (use Markdown code blocks)
     ((and beg end)
      (format "File: %s (Lines %d-%d)\n```%s\n%s\n```\n"
              (or file (buffer-name))
              (line-number-at-pos beg)
              (line-number-at-pos end)
              lang
              (buffer-substring-no-properties beg end)))

     ;; Case 2: No region, but associated to a file
     (file
      (format "File: %s\n" file))

     ;; Case 3: No region, not a file
     (t nil))))

(defun ce/vterm-send-region (beg end target-buffer prompt-text)
  "Send PROMPT-TEXT and optional buffer context to a vterm buffer.
If a region is active, sends the region text and line numbers.
If no region is active, in an org-mode file, and on a subtree heading, sends
only the subtree content.
If no region is active but the buffer visits a file, sends the file name.
If no region is active and no file is visited, sends only the prompt.
File paths are made relative to the target vterm's current directory."
  (interactive
   (let* ((has-region (use-region-p))
          (is-org-heading (and (not has-region)
                               (derived-mode-p 'org-mode)
                               (org-at-heading-p))))
     (list (when has-region (region-beginning))
           (when has-region (region-end))
           (ce/prompt-vterm-buffer)
           (if is-org-heading "" (read-string "Prompt: ")))))

  (unless (get-buffer target-buffer)
    (error "Buffer %s not found." target-buffer))

  (let* ((vterm-dir (with-current-buffer target-buffer default-directory))
         (context (ce/get-buffer-context beg end vterm-dir))
         (prompt-empty (string-empty-p (or prompt-text "")))
         (payload
          (cond
           ((and context (not prompt-empty)) (format "%s\n\n%s" prompt-text context))
           (context context)
           (t (format "%s\n" prompt-text)))))

    (unless (or (null payload) (string-empty-p (string-trim payload)))
      (ce/vterm-send-string payload target-buffer nil))))

(defun ce/org-send-region (beg end target-buffer-name &optional path-anchor)
  "Append the context of the current buffer to an open Org buffer.
BEG and END define the region if active. TARGET-BUFFER-NAME is the
destination Org buffer. If PATH-ANCHOR is non-nil, truncates the
source file path up to it to use as =target-dir=. Otherwise,
=target-dir= is passed as nil to default to the absolute path."
  (interactive
   (let* ((has-region (use-region-p))
          (beg (when has-region (region-beginning)))
          (end (when has-region (region-end)))
          ;; Collect all open buffers running org-mode
          (org-buffers (delq nil (mapcar (lambda (b)
                                           (when (with-current-buffer b
                                                   (derived-mode-p 'org-mode))
                                             (buffer-name b)))
                                         (buffer-list))))
          (target (if org-buffers
                      (completing-read "Target Org buffer: " org-buffers nil t)
                    (error "No Org buffers are currently open"))))
     ;; Pass nil for path-anchor by default when called interactively
     (list beg end target nil)))

  (let* ((target-buffer (get-buffer target-buffer-name))
         (source-path (or (buffer-file-name) default-directory))
         ;; If path-anchor is given and matches, extract up to it. Otherwise nil.
         (target-dir (when (and path-anchor
                                (string-match (format "^.*%s" (regexp-quote path-anchor)) source-path))
                       (file-name-as-directory (match-string 0 source-path))))
         (context (ce/get-buffer-context beg end target-dir)))
    (if (not context)
        (message "No context found to send.")
      (with-current-buffer target-buffer
        (save-excursion
          (goto-char (point-max))
          ;; Ensure we start on a new line
          (unless (bolp) (insert "\n"))
          ;; Add a blank line separator and insert the context
          (insert "\n" context "\n")))
      (message "Appended context to %s" target-buffer-name))))

(defvar ce/run-cli-output-mode 'markdown-mode
  "Default major mode for output buffers in `ce/run-cli-with-context`.")

(defun ce/run-cli-with-context (command-string beg end user-prompt)
  "Run COMMAND-STRING asynchronously, passing buffer context and USER-PROMPT.
COMMAND-STRING can include arguments. BEG and END define the region if active."
  (interactive
   (let* ((has-region (use-region-p))
          (beg (when has-region (region-beginning)))
          (end (when has-region (region-end)))
          (cmd (read-string "CLI command (with args): "))
          (prompt (read-string "Prompt: ")))
     (list cmd beg end prompt)))

  (let* ((context (ce/get-buffer-context beg end))
         (final-text (concat (or context "")
                             (when (and context (not (string-empty-p user-prompt)))
                               "\n\n--- Prompt ---\n")
                             user-prompt))
         (cmd-parts (split-string-and-unquote command-string))
         (program (car cmd-parts))
         (cli-args (cdr cmd-parts))
         (buf-base-name (format "*%s-output*" program))
         ;; Generate a guaranteed unique buffer (e.g., *program-output*<2>)
         (out-buf (generate-new-buffer buf-base-name)))

    (if (string-empty-p (string-trim final-text))
        (user-error "Nothing to send: context and prompt are both empty")

      (with-current-buffer out-buf
        (funcall ce/run-cli-output-mode)
        ;; erase-buffer is no longer needed since it's a fresh buffer
        (insert final-text "\n\n--- Output ---\n"))
      (display-buffer out-buf)

      (apply #'start-process
             ;; Make the process name unique by tying it to the buffer name
             (format "%s-process" (buffer-name out-buf))
             out-buf
             program
             (append cli-args (list final-text)))

      (message "Started %s in the background..." program))))

(defun ce/open-latest-buffers ()
  "Use three frames (current + 2 new), split them, and display the 6 latest visited buffers."
  (interactive)
  (require 'seq)
  ;; Get the 6 most recent buffers, ignoring hidden ones
  (let* ((bufs (seq-take (seq-filter (lambda (b)
                                       (not (string-prefix-p " " (buffer-name b))))
                                     (buffer-list))
                         6))
         ;; Use the current frame, plus create two new ones
         (frames (list (selected-frame) (make-frame) (make-frame))))

    ;; Setup all three frames
    (cl-loop for frame in frames
             for i from 0 by 2 do
             (select-frame frame)
             (delete-other-windows)
             (let ((b1 (nth i bufs))
                   (b2 (nth (1+ i) bufs)))
               (when b1 (set-window-buffer (selected-window) b1))
               (when b2 (set-window-buffer (split-window-right) b2))))))

(defun ce/bibtex-extract-arxiv-id (entry)
  "Extract arXiv ID (with optional version) from BibTeX ENTRY alist.
Checks URL first (to preserve specific versions like v1), then eprint,
then journal."
  (let ((url (cdr (assoc-string "url" entry t)))
        (eprint (cdr (assoc-string "eprint" entry t)))
        (archive-prefix (cdr (assoc-string "archiveprefix" entry t)))
        (journal (cdr (assoc-string "journal" entry t)))
        (arxiv-id-re "\\([0-9]\\{4\\}\\.[0-9]\\{4,5\\}\\(?:v[0-9]+\\)?\\|[a-zA-Z-]+/[0-9]\\{7\\}\\(?:v[0-9]+\\)?\\)"))
    (cond
     ;; 1. Check URL: if it points to arXiv, extract ID (including version if specified)
     ((and url (string-match (concat "arxiv\\.org/\\(?:abs\\|pdf\\)/" arxiv-id-re) url))
      (match-string 1 url))
     ;; 2. Check eprint field
     ((and eprint
           (or (null archive-prefix)
               (string-match-p "arxiv" (downcase archive-prefix))
               (string-match-p (concat "^" arxiv-id-re "$") eprint)))
      (if (string-match arxiv-id-re eprint)
          (match-string 1 eprint)
        eprint))
     ;; 3. Check journal field (e.g. "arXiv preprint arXiv:1502.00192")
     ((and journal
           (string-match-p "arxiv" (downcase journal))
           (string-match arxiv-id-re journal))
      (match-string 1 journal))
     ;; Not an arXiv paper
     (t nil))))

(defun ce/bibtex-download-arxiv-pdfs (bibfile pdf-dir)
  "Download missing arXiv PDFs for entries in BIBFILE into PDF-DIR.
Each PDF is named <key>.pdf where <key> is the BibTeX citation key.
If <key>.pdf already exists in PDF-DIR, it is skipped.
If an entry is not on arXiv, it is skipped.
Returns a plist with (:downloaded ... :skipped-exists ... :skipped-not-arxiv ... :failed ...)."
  (interactive "fBibTeX file: \nDDirectory for PDFs: ")
  (require 'parsebib)
  (require 'org-ref-arxiv)
  (let* ((bibfile (expand-file-name bibfile))
         (pdf-dir (file-name-as-directory (expand-file-name pdf-dir)))
         (entries (parsebib-parse bibfile))
         downloaded skipped-exists skipped-not-arxiv failed)
    (unless (file-directory-p pdf-dir)
      (make-directory pdf-dir t))
    (maphash
     (lambda (key entry)
       (let ((pdf-path (concat pdf-dir key ".pdf")))
         (cond
          ;; 1. Check if PDF already exists
          ((file-exists-p pdf-path)
           (message "Skipping %s: %s.pdf already exists" key key)
           (push key skipped-exists))
          ;; 2. Check if paper is on arXiv
          (t
           (let ((arxiv-id (ce/bibtex-extract-arxiv-id entry)))
             (if (not arxiv-id)
                 (progn
                   (message "Skipping %s: Not on arXiv" key)
                   (push key skipped-not-arxiv))
               (message "Downloading %s (arXiv: %s) -> %s..." key arxiv-id pdf-path)
               (condition-case err
                   (progn
                     (arxiv-get-pdf arxiv-id pdf-path)
                     (if (file-exists-p pdf-path)
                         (progn
                           (message "Successfully downloaded %s -> %s" key pdf-path)
                           (push key downloaded))
                       (message "Failed to download %s (arXiv: %s)" key arxiv-id)
                       (push key failed)))
                 (error
                  (message "Error downloading %s: %s" key (error-message-string err))
                  (push key failed)))))))))
     entries)
    (list :downloaded (nreverse downloaded)
          :skipped-exists (nreverse skipped-exists)
          :skipped-not-arxiv (nreverse skipped-not-arxiv)
          :failed (nreverse failed))))
