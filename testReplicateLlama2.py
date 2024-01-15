import replicate


def run_llama() -> None:

    prompt: str = \
"""
For context, I am the project lead and the point person of many software development projects.
My name is Adrian Frei.
I need an autoresponse email whenever I am not available. Do not start with the subject field, just the body.
Also, do not include your communication to me in your response. 
Please tell them I will be back in the 20th of December and that at the moment there is no one back in the office to answer their request.
Do not mention that the email is an autoresponder.

This is an email from one of the clients. Please formulate a reponse:
Hello Adrian,

I need some support on one of the projects we have deployed together. Are you by any chance free today?

Regards,
Thomas
"""
    system_prompt: str = "you are a personal assistant who responds to emails professionally"

    output = replicate.run(
        # "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",     # Llama2 70-billion parameter chat model
        "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0",        # Llama2 7-billion parameter chat model
        # "meta/llama-2-7b:73001d654114dad81ec65da3b834e2f691af1e1526453189b7bf36fb3f32d0f9",             # Llama2 7-billion parameter base model
        input={
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": 0.75
               }
    )

    # The meta/llama-2-chat model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    for item in output:
        # print(item)
        print(item, end="")


def main() -> None:
    run_llama()


if __name__ == "__main__":
    main()