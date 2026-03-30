import os
import sys
import subprocess
import traceback
from datetime import datetime


def _parse_seed_count(argv):
    seed_count = int(os.environ.get("THERMO_DEMO_SEED_COUNT", "1"))
    if "--seed-count" in argv:
        try:
            idx = argv.index("--seed-count")
            seed_count = int(argv[idx + 1])
        except Exception:
            seed_count = seed_count
    return seed_count


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    seed_count = _parse_seed_count(sys.argv)
    seeds = list(range(seed_count))

    demos = [
        "demo_xor.py",
        "demo_alu.py",
        "demo_alu_generalization.py",
        "demo_logic_gates.py",
        "demo_noise.py",
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    report_name = f"rapport_tests_{timestamp}.txt"
    report_path = os.path.join(base_dir, report_name)

    print(f"Lancement de {len(demos)} démo(s)...")
    print(f"Génération du rapport dans : {report_name}")

    with open(report_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Date d'exécution : {timestamp}\n")
        log_file.write(f"Seeds : {seeds}\n")
        log_file.write("\n")

        for demo in demos:
            demo_path = os.path.join(base_dir, demo)
            if not os.path.isfile(demo_path):
                continue

            log_file.write("\n" + "#" * 70 + "\n")
            log_file.write(f"DÉMARRAGE DU TEST : {demo}\n")
            log_file.write("#" * 70 + "\n\n")
            log_file.flush()

            for seed in seeds:
                log_file.write("\n" + "-" * 70 + "\n")
                log_file.write(f"Seed = {seed}\n")
                log_file.write("-" * 70 + "\n")
                log_file.flush()

                env = os.environ.copy()
                env["THERMO_DEMO_SEED"] = str(seed)

                try:
                    result = subprocess.run(
                        [sys.executable, demo_path],
                        capture_output=True,
                        text=True,
                        check=False,
                        cwd=base_dir,
                        env=env,
                    )

                    if result.stdout:
                        log_file.write(result.stdout)

                    if result.stderr:
                        log_file.write("\n--- ERREURS / WARNINGS ---\n")
                        log_file.write(result.stderr)

                    if result.returncode == 0:
                        log_file.write("\n\n[STATUT DU TEST] : SUCCÈS\n")
                    else:
                        log_file.write(f"\n\n[STATUT DU TEST] : ÉCHEC (Code {result.returncode})\\n")
                except Exception as e:
                    log_file.write(f"\nErreur critique lors de l'exécution : {e}\n")
                    log_file.write(traceback.format_exc())

    print(f"\nTous les tests sont terminés ! Log enregistré sous : {report_name}")


if __name__ == "__main__":
    main()
