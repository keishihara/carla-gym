# Leaderboard 2.0統合ガイド

CarlaGymにLeaderboard 2.0のシナリオシステムが統合されました。本ドキュメントでは、統合された機能の使用方法と技術的詳細について説明します。

## 概要

### 統合された機能

- **25種類のシナリオタイプ**: Accident、BlockedIntersection、ConstructionObstacle等
- **自動ルート発見**: leaderboard_2_0のディレクトリ構造から自動的にXMLルートを発見
- **シナリオマネージャー**: scenario_manager_local.pyのスタック検出機能を統合
- **TaskConfig拡張**: 既存のTaskConfigAPIにleaderboard_2_0サポートを追加

### アーキテクチャ

```
carla_gym/
├── external/srunner/           # leaderboard_2_0のsrunnerパッケージ
├── envs/
│   ├── leaderboard_scenario_loader.py  # XMLルート自動発見・読み込み
│   └── task_config.py          # leaderboard_2_0統合メソッド追加
├── engine/actors/scenario_actor/
│   └── scenario_actor_handler.py       # スタック検出機能統合
tests/
└── test_leaderboard_scenarios.py       # 統合テストスイート
```

## セットアップ

### 前提条件

1. CARLA 0.9.15が正しくインストールされていること
2. leaderboard_2_0データがcarla_benchmarks/leaderboard_2_0/dataに配置されていること
3. 必要なPythonパッケージ（py_trees、numpy等）がインストールされていること

### 依存関係の確認

```python
# 必要なパッケージの確認
import py_trees
import carla
import xml.etree.ElementTree as ET
from carla_gym.envs.leaderboard_scenario_loader import create_leaderboard_loader
```

## 基本的な使用方法

### 1. LeaderboardScenarioLoaderの使用

```python
from carla_gym.envs.leaderboard_scenario_loader import create_leaderboard_loader

# ローダーの作成（自動パス検出）
loader = create_leaderboard_loader()

# または明示的にパス指定
loader = create_leaderboard_loader("/path/to/leaderboard_2_0/data")

# 利用可能なシナリオタイプの確認
scenario_types = loader.get_available_scenario_types()
print(f"利用可能なシナリオ: {scenario_types}")

# 利用可能な町の確認
towns = loader.get_available_towns()
print(f"利用可能な町: {towns}")

# 統計情報の表示
stats = loader.get_stats()
print(f"統計: {stats}")
```

### 2. ランダムルートの取得

```python
# 特定のシナリオタイプのランダムルート
accident_route = loader.get_random_route("Accident")
print(f"事故シナリオルート: {accident_route.route_id}")

# 特定の町のランダムルート
town13_route = loader.get_random_route(town="Town13")
print(f"Town13のルート: {town13_route.scenario_type}")

# 完全ランダムルート
random_route = loader.get_random_route()
```

### 3. TaskConfigとの統合

```python
from carla_gym.envs.task_config import TaskConfig

# leaderboard_2_0からTaskConfigを作成
task_config = TaskConfig.from_leaderboard_2_0(
    scenario_type="Accident",
    town="Town13",
    num_npc_vehicles=5,
    num_npc_walkers=3,
    weather="ClearNoon"
)

# 複数のTaskConfigをサンプリング
task_configs = TaskConfig.sample_leaderboard_scenarios(
    n=10,
    scenario_types=["Accident", "BlockedIntersection"],
    towns=["Town13"],
    num_npc_vehicles=(0, 10),
    num_npc_walkers=(0, 5),
    weathers=["ClearNoon", "CloudyNoon", "WetNoon"]
)
```

### 4. CarlaEnvでの使用

```python
from carla_gym.envs.carla_env import CarlaEnv
from carla_gym.envs.task_config import TaskConfig

# TaskConfigを作成
task_config = TaskConfig.from_leaderboard_2_0(
    scenario_type="DynamicObjectCrossing",
    town="Town13"
)

# CarlaEnvを初期化
env = CarlaEnv(
    map_name="Town13",
    task_config=task_config
)

# 環境をリセット（シナリオが自動的にロードされる）
obs = env.reset()
```

## APIリファレンス

### LeaderboardScenarioLoader

#### クラスメソッド

| メソッド | 説明 | 引数 | 戻り値 |
|---------|------|------|--------|
| `__init__(data_path)` | ローダーを初期化 | `data_path`: leaderboard_2_0/dataのパス | なし |
| `get_available_scenario_types(town=None)` | 利用可能なシナリオタイプ一覧 | `town`: 町でフィルター（オプション） | `List[str]` |
| `get_available_towns()` | 利用可能な町一覧 | なし | `List[str]` |
| `get_routes_by_type(scenario_type, town=None)` | 指定タイプのルート一覧 | `scenario_type`: シナリオタイプ<br>`town`: 町（オプション） | `List[LeaderboardRoute]` |
| `get_random_route(scenario_type=None, town=None)` | ランダムルート取得 | `scenario_type`: シナリオタイプ（オプション）<br>`town`: 町（オプション） | `LeaderboardRoute` or `None` |
| `create_task_config(route, **kwargs)` | ルートからTaskConfig作成 | `route`: LeaderboardRoute<br>`**kwargs`: TaskConfig追加パラメータ | `TaskConfig` |
| `get_stats()` | 統計情報取得 | なし | `Dict[str, int]` |

### TaskConfig拡張メソッド

#### 新規クラスメソッド

| メソッド | 説明 | 引数 | 戻り値 |
|---------|------|------|--------|
| `from_leaderboard_2_0(**kwargs)` | leaderboard_2_0からTaskConfig作成 | `scenario_type`: シナリオタイプ<br>`town`: 町<br>`num_npc_vehicles`: NPC車両数<br>等 | `TaskConfig` or `None` |
| `sample_leaderboard_scenarios(**kwargs)` | 複数TaskConfigサンプリング | `n`: 作成数<br>`scenario_types`: シナリオタイプリスト<br>等 | `List[TaskConfig]` |

### ScenarioActorHandler拡張

#### スタック検出機能

| 定数 | 値 | 説明 |
|------|---|------|
| `STUCK_DISTANCE_THRESHOLD_METERS` | 0.5 | スタック判定距離閾値（メートル） |
| `STUCK_DURATION_THRESHOLD_SECONDS` | 10.0 | スタック判定時間閾値（秒） |
| `MAX_CONSECUTIVE_STUCK_TICKS` | 50 | 最大連続スタックティック数 |

#### 新規メソッド

- `is_scenario_aborted_due_to_stuck`: スタック検出によるシナリオ中断フラグ

## 対応シナリオタイプ

| シナリオタイプ | 説明 | 特徴 |
|---------------|------|------|
| Accident | 事故シナリオ | 静的障害物による事故再現 |
| AccidentTwoWays | 双方向事故 | 対向車線を含む事故シナリオ |
| BlockedIntersection | 交差点ブロック | 交差点での車両停止シナリオ |
| ConstructionObstacle | 建設障害物 | 工事現場での障害物回避 |
| ConstructionObstacleTwoWays | 双方向建設障害物 | 対向車線を含む工事現場 |
| ControlLoss | 車両制御喪失 | 制御不能車両への対処 |
| DynamicObjectCrossing | 動的物体横断 | 歩行者・自転車等の横断 |
| EnterActorFlow | 車両流入 | 交通流への合流シナリオ |
| HardBreakRoute | 急ブレーキルート | 緊急停止が必要なシナリオ |
| HazardAtSideLane | 側道危険 | 側道での危険物対処 |
| InvadingTurn | 侵入ターン | 不正な車線変更・ターン |
| MergerIntoSlowTraffic | 低速交通合流 | 渋滞への合流シナリオ |
| NonSignalizedJunctionLeftTurn | 無信号左折 | 信号のない交差点での左折 |
| NonSignalizedJunctionRightTurn | 無信号右折 | 信号のない交差点での右折 |
| OppositeVehicleRunningRedLight | 対向車赤信号無視 | 対向車の信号無視シナリオ |
| OppositeVehicleTakingPriority | 対向車優先 | 対向車の優先権主張 |
| OtherLeadingVehicle | 先行車追従 | 先行車両への追従シナリオ |
| ParkingCutIn | 駐車車両割り込み | 駐車車両からの割り込み |
| ParkingExit | 駐車場退場 | 駐車場からの退場シナリオ |
| PedestrianCrossing | 歩行者横断 | 歩行者横断歩道シナリオ |
| SignalizedJunctionLeftTurn | 信号左折 | 信号機のある交差点での左折 |
| SignalizedJunctionRightTurn | 信号右折 | 信号機のある交差点での右折 |
| VehicleOpensDoor | 車両ドア開放 | 駐車車両のドア開放シナリオ |
| YieldToEmergencyVehicle | 緊急車両への譲歩 | 救急車・消防車等への道譲り |

## パフォーマンス最適化

### メモリ使用量

- XMLルートキャッシュ：初回読み込み時に全XMLをメモリにキャッシュ
- 大規模データセット対応：遅延読み込みでメモリ効率を最適化

### 実行速度

- ルート選択：O(1)のランダムアクセス
- XMLパース：初回のみ、以降はキャッシュ使用
- シナリオ構築：~1Hz頻度で動的構築

## トラブルシューティング

### よくある問題

#### 1. leaderboard_2_0データが見つからない

```
FileNotFoundError: Could not find leaderboard_2_0 data directory
```

**解決方法:**
```python
# 明示的にパスを指定
loader = create_leaderboard_loader("/full/path/to/leaderboard_2_0/data")
```

#### 2. XMLパースエラー

```
ET.ParseError: XML parse error
```

**解決方法:**
- XMLファイルの形式を確認
- ファイルのエンコーディングがUTF-8であることを確認
- 破損したXMLファイルを除外

#### 3. シナリオが見つからない

```
logger.warning("No routes found for scenario_type=..., town=...")
```

**解決方法:**
```python
# 利用可能なシナリオタイプを確認
print(loader.get_available_scenario_types())
print(loader.get_available_towns())

# 統計情報で詳細確認
print(loader.get_stats())
```

#### 4. スタック検出の誤判定

**調整方法:**
```python
from carla_gym.engine.actors.scenario_actor.scenario_actor_handler import ScenarioActorHandler

# 閾値調整（必要に応じて）
handler = ScenarioActorHandler(client)
handler.STUCK_DISTANCE_THRESHOLD_METERS = 1.0  # デフォルト: 0.5
handler.STUCK_DURATION_THRESHOLD_SECONDS = 15.0  # デフォルト: 10.0
```

### デバッグのヒント

#### ログ出力の有効化

```python
import logging
logging.getLogger('carla_gym.envs.leaderboard_scenario_loader').setLevel(logging.DEBUG)
logging.getLogger('carla_gym.engine.actors.scenario_actor').setLevel(logging.DEBUG)
```

#### XMLファイルの詳細確認

```python
# XMLの内容確認
route = loader._parse_xml_route(Path("route.xml"), "Town13", "Accident")
print(f"Waypoints: {len(route.waypoints)}")
print(f"Scenarios: {len(route.scenarios)}")
print(f"Weather: {route.weather}")
```

## 制限事項と注意点

### 現在の制限事項

1. **Python 3.8+必須**: srunnerの依存関係によりPython 3.8以上が必要
2. **メモリ使用量**: 大規模データセット（>1000ルート）では初期化に時間がかかる場合
3. **CARLA バージョン**: CARLA 0.9.15専用（他バージョンでの動作保証なし）

### 移植時の変更点

1. **インポートパス変更**: 
   - 旧: `from leaderboard.scenarios import ...`
   - 新: `from carla_gym.external.srunner.scenarios import ...`

2. **スタック検出統合**: scenario_manager_local.pyの機能をScenarioActorHandlerに統合

3. **エラーハンドリング強化**: XMLパースエラーやファイル不存在エラーの処理改善

### 今後の開発予定

- [ ] より多くのCARLAバージョンへの対応
- [ ] パフォーマンス最適化（大規模データセット対応）
- [ ] カスタムシナリオタイプのサポート
- [ ] GUI統合（シナリオ選択インターフェース）

## サンプルコード

### 完全な使用例

```python
#!/usr/bin/env python3
"""
Leaderboard 2.0統合の完全な使用例
"""

from carla_gym.envs.carla_env import CarlaEnv
from carla_gym.envs.task_config import TaskConfig
from carla_gym.envs.leaderboard_scenario_loader import create_leaderboard_loader

def main():
    # 1. ローダーの作成と確認
    print("=== Leaderboard 2.0ローダー初期化 ===")
    loader = create_leaderboard_loader()
    
    stats = loader.get_stats()
    print(f"総ルート数: {stats['total_routes']}")
    print(f"シナリオタイプ数: {stats['scenario_types']}")
    print(f"利用可能な町: {loader.get_available_towns()}")
    
    # 2. 特定シナリオでのTaskConfig作成
    print("\\n=== TaskConfig作成 ===")
    task_config = TaskConfig.from_leaderboard_2_0(
        scenario_type="DynamicObjectCrossing",
        town="Town13",
        num_npc_vehicles=8,
        num_npc_walkers=5,
        weather="ClearNoon"
    )
    
    if task_config:
        print(f"作成成功: {task_config.route_id}")
        print(f"マップ: {task_config.map_name}")
        print(f"シナリオ数: {len(task_config.scenarios)}")
    
    # 3. 複数シナリオのサンプリング
    print("\\n=== 複数シナリオサンプリング ===")
    scenarios = TaskConfig.sample_leaderboard_scenarios(
        n=5,
        scenario_types=["Accident", "BlockedIntersection", "PedestrianCrossing"],
        towns=["Town13"],
        weathers=["ClearNoon", "CloudyNoon"]
    )
    
    for i, scenario in enumerate(scenarios):
        route = loader.get_routes_by_type(scenario.scenarios[0].type)[0]
        print(f"シナリオ {i+1}: {route.scenario_type} ({scenario.weather})")
    
    # 4. CarlaEnvでの実行（オプション）
    print("\\n=== CarlaEnv統合テスト ===")
    if task_config:
        try:
            env = CarlaEnv(
                map_name=task_config.map_name,
                task_config=task_config
            )
            print("CarlaEnv初期化成功")
            # obs = env.reset()  # 実際のCarlaサーバーが必要
            # print("環境リセット成功")
        except Exception as e:
            print(f"CarlaEnv初期化失敗（Carlaサーバー未起動?）: {e}")

if __name__ == "__main__":
    main()
```

## 関連リンク

- [CARLA公式ドキュメント](https://carla.readthedocs.io/)
- [ScenarioRunner GitHub](https://github.com/carla-simulator/scenario_runner)
- [CarlaGym リポジトリ](https://github.com/YourOrg/carla-gym)
- [Leaderboard 2.0 仕様](https://leaderboard.carla.org/)

---

**更新履歴:**
- 2024-12-XX: 初版作成
- 今後のアップデートに応じて随時更新予定