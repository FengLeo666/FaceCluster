📸 Intelligent Photo Album System

This project implements an intelligent photo album system based on face detection, alignment, and clustering.
It automatically organizes large collections of photos into albums of individuals, allowing efficient browsing, management, and sharing.

✨ Key Features

Front-end (Web & PWA)

Session list management (create, rename, merge, append).

Batch uploads (multiple files or ZIP).

Interactive clustering result naming and merging.

Album browsing with grid view, filters, and statistics.

PWA support: responsive layout, offline fallback, touch-optimized.

Back-end Services

Session management with state transitions (pending, processing, completed, failed).

Image pipeline: preprocessing → face detection & alignment → feature extraction.

Clustering & album construction with representative thumbnails.

Conflict resolution with majority voting and last-write-wins policy.

On-demand ZIP packaging and temporary share links.

Storage & Metadata

Organized session-based directory structure (raw, albums, thumbnails).

Atomic metadata updates (session.json) with fault tolerance.

Optional feature vector caching for incremental processing.

Collaboration

Roles: session owner & collaborators (editable).

Human-readable conflict handling during album naming/merging.

Real-time updates across clients.

🚀 Typical Workflow

Upload → Preprocessing → Face detection & feature extraction → Clustering → Album construction & thumbnails → Distribution & sync → Front-end confirmation & renaming.
