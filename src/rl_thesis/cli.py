from os import name
import typer

app = typer.Typer(name="rl_thesis", help="My Survival RL Thesis Codebase")

@app.command()
def demo():
    from rl_thesis.demo.demo import demo
    demo()

if __name__ == "__main__":
    app()