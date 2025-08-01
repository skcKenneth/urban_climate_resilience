# GitHub Actions Troubleshooting Guide

## Overview
This guide helps troubleshoot common issues with the Urban Climate Resilience Analysis GitHub Actions workflow.

## Quick Checks

### 1. Verify Setup Locally
Run the verification script to check your setup:
```bash
python verify_setup.py
```

### 2. Test Locally Before Pushing
Run the quick test locally:
```bash
export QUICK_MODE=true
export MPLBACKEND=Agg
python test_quick_run.py
```

## Common Issues and Solutions

### Issue: Import Errors
**Symptoms**: `ModuleNotFoundError` or `ImportError` in GitHub Actions logs

**Solutions**:
1. Ensure all dependencies are in `requirements.txt`
2. Check that `PYTHONPATH` is set correctly in the workflow
3. Verify all `__init__.py` files exist in subdirectories

### Issue: Timeout Errors
**Symptoms**: Job cancelled due to timeout

**Solutions**:
1. Increase timeout values in workflow file
2. Use `QUICK_MODE=true` for faster testing
3. Reduce `N_SAMPLES` and `SIMULATION_DAYS` environment variables

### Issue: No Output Files Generated
**Symptoms**: No PNG files or results found

**Solutions**:
1. Check that output directory is created
2. Verify matplotlib backend is set to 'Agg'
3. Check for errors in visualization code
4. Ensure proper file permissions

### Issue: Memory Errors
**Symptoms**: Process killed or out of memory errors

**Solutions**:
1. Reduce population size in quick mode
2. Limit parallel operations
3. Use smaller sample sizes

## Environment Variables

The workflow uses these environment variables:

- `QUICK_MODE`: Set to 'true' for faster testing
- `MPLBACKEND`: Should be 'Agg' for headless environments
- `PYTHONUNBUFFERED`: Set to '1' for real-time output
- `MAX_TIME`: Maximum time in seconds for analysis
- `N_SAMPLES`: Number of samples for sensitivity analysis
- `SIMULATION_DAYS`: Number of days to simulate
- `OMP_NUM_THREADS`: Number of threads for OpenMP

## Debugging Steps

1. **Check GitHub Actions Logs**:
   - Look for the first error in the logs
   - Check the "Set up Python" step for installation issues
   - Review "Check imports" step for missing modules

2. **Run Minimal Test**:
   ```yaml
   - name: Debug imports
     run: |
       python -c "import sys; print(sys.path)"
       python -c "import models; print(models.__file__)"
   ```

3. **Add Debug Output**:
   - Add `echo` statements in workflow
   - Use `ls -la` to check file existence
   - Print environment variables

4. **Test Incrementally**:
   - Start with just the test job
   - Add scenarios one at a time
   - Use `fail-fast: false` to see all failures

## Workflow Configuration Tips

1. **Dependencies**:
   ```yaml
   - name: Install dependencies
     run: |
       python -m pip install --upgrade pip
       pip install wheel setuptools
       pip install -r requirements.txt
   ```

2. **Python Path**:
   ```yaml
   - name: Run analysis
     run: |
       export PYTHONPATH="${PYTHONPATH}:${PWD}"
       python main.py
   ```

3. **Error Handling**:
   ```yaml
   - name: Check results
     if: always()
     run: |
       ls -la results/ || echo "No results found"
   ```

## Testing Changes

Before pushing changes:

1. Run `python verify_setup.py`
2. Run `python test_quick_run.py`
3. Check for any new dependencies
4. Test with minimal parameters first

## Getting Help

If issues persist:
1. Check the workflow run logs in GitHub Actions tab
2. Look for similar issues in GitHub Issues
3. Run with increased verbosity/logging
4. Test with a minimal reproducible example