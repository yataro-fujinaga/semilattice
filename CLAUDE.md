# Semilattice (sl)

ファイルツリーの上に乗る意味的関係レイヤー。Rust製CLI。

## セットアップ

```bash
# ビルド
cargo build

# iCloud同期回避
mv target target.nosync && ln -s target.nosync target

# グローバルインストール（~/.cargo/bin/sl）
cargo install --path .
```

初回の `sl query` 実行時に multilingual-e5-small モデルが自動ダウンロードされる（約100MB）。

## 使い方

```bash
# プロジェクトで初期化
cd your-project
sl init

# 関係を記録
sl add "ログインバグ修正" src/auth.rs src/handlers.rs tests/auth_test.rs

# 意味的に検索（ベクトル検索 + グラフ探索）
sl query "認証"

# コンテキスト一覧
sl contexts

# 特定コンテキストの詳細
sl show "ログイン"
```

## 設計ドキュメント

- `DESIGN.md` — 思想・価値・アーキテクチャ・シナリオの全体設計
- `CLAUDE_TEMPLATE.md` — sl を導入するプロジェクトの CLAUDE.md に貼るテンプレート

## 技術スタック

- 言語: Rust
- 埋め込みモデル: intfloat/multilingual-e5-small（candle, pure Rust, ONNX不要）
- ストレージ: SQLite（.sl/relations.db）
- 検索: ベクトル類似度で入口を見つけ、共有ファイル経由でグラフ探索（BFS + ホップベース減衰）

## プロジェクト構造

```
src/
  main.rs        CLI（init, add, query, contexts, show）
  db.rs          SQLite スキーマ定義
  store.rs       データ操作（add_context, query_graph, list, show）
  embedding.rs   ベクトル化（multilingual-e5-small, コサイン類似度）
```

## sl — AIエージェント向け指示

このプロジェクト自身でも `sl` を使っている。

### 作業開始時

```bash
sl query "タスクの内容"
```

### 作業完了時

```bash
sl add "作業内容" 変更したファイル...
```
