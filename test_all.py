#!/usr/bin/env python
"""
Test script for LOTUS project components.
This script runs tests for each module individually to ensure they're working correctly.
"""

import os
import sys
import subprocess
import time

def print_header(module_name):
    """Print a nice header for each test."""
    border = "=" * 80
    print(f"\n{border}")
    print(f"Testing {module_name}".center(80))
    print(f"{border}\n")

def run_test(command, module_name):
    """Run a test command and capture the output."""
    print_header(module_name)
    try:
        process = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(process.stdout)
        if process.stderr:
            print(f"STDERR: {process.stderr}")
        print(f"\n✅ {module_name} test completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        print(f"\n❌ {module_name} test failed.")
        return False

def main():
    """Run all tests sequentially."""
    # Ensure test sequence file exists
    if not os.path.exists("test_sequence.fasta"):
        print("Creating test sequence file...")
        with open("test_sequence.fasta", "w") as f:
            f.write(""">test_cas9
MLTYDVHNGTCKPCLPAGCVRECNGGRVDSVAAAEHFRGCTRIKGTLEITLRASGGKYTP
HTHLYTTSVHNCNTISVLESSLGEIREIDGSLKVVRSYPLVSLMFLKKLRRIGGLNSDAK
TPSLYISNNPNLEMLWDWSVHKPIEITRGKLYIHFNSKLCYNHILELKNMTRDPVSTFDT
IEVSPESNGDQAPCLQDVLELKVSTLKQRAALLTWKMYCGIDTRNILGYSIYYIAAEHNV
TLYGQRDACSDTWSVKDIPMDTARFTETKTTAIAMNGLRDPCEDMQPFFSPISPLVPFKR
YAAYVKTYTTKQDKKGAQSPIIYFKTLPDSPSPPLGLTVELRTPHSVQIKWQPPALPNGT
ISLYHVEIQANSYNRRTILADNLNYCSNRMYYAAEQERIDSIDFENFIQNHVYIRNEKVK
TPKSGGKVKRAIEKEINSMLVILSGYNKPKQGLKYYNSTDTEGYVKSLYFELDGSARSLV
VEEMRHYTWYTVNLWACRQKQDYEPEDNYDKTWCSGRSFNTFRTLELPNADVVQDVRVEV
ITSNKTLPELNVTWKPPENPNGFAIAYFVQHSRIVDNNQAQDVGLQRCITATDYEANGRG
YTLRNMAPGNYSVRVTPVTVSGAGNVSAHVYAFIPDRDSEKGFLWAWGVAAGVLLLVLLA
GGAWYARRALPSPEGNKLFATVNPEYVSTVYVPDEWELPRSSIEFIRELGQGSFGMVYEG
IAKSIEKGKPETRCAVKTVNEHATDRERIEFLNEASVMKAFDTYHVVRLLGVVSRGQPTL
VVMELMECGDLKTYLRSHRPDADSSLPKKDDNAPPTLQNILQMAIEIADGMAYLAAKKYV
HRDLAARNCMVAGDLTVKVGDFGMTRDIYETDYYRKGTKGLLPVRWMSPESLKDGVFSSN
SDVWSYGVVLWEMATLAMQPYQGLSNEQVVRYVVEGGVMERPEHCPDRLYELMRACWAHR
ANTRPSFLQLVADLAPSAQPYFRHRSFFHSPQGQEMYALMRTTVEEEAECAEVNVGAVAT
GSGSNLFGVSGRLATWVRELSSLHSRADDDAAAQPLQPARAHKGPNGVLHDPLDPLDDTT
GC""")
    
    # Tests to run
    tests = [
        ("python -m utils.mmseqs_split", "MMseqs2 Split"),
        ("python -m generator.embed_sequences", "ESM-2 Embedding"),
        ("python -m generator.models", "Flow Matching Network"),
        ("python -m generator.solver", "ODE Solvers"),
        ("python -m generator.decode", "Decoder"),
        ("python -m generator.validate", "Validation"),
        ("python -m generator.downstream", "Downstream Evaluation"),
        ("python -m generator.train_decoder --test", "Decoder Training")
    ]
    
    results = {}
    
    # Run each test
    for command, name in tests:
        print(f"\nRunning test for {name}...")
        success = run_test(command, name)
        results[name] = success
        time.sleep(1)  # Small delay between tests
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("Test Summary".center(80))
    print("=" * 80)
    
    all_passed = True
    for name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! The LOTUS system is ready to run.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before running the full experiment.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 