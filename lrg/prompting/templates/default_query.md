<system> {% include system_prompt %}

{% for turn_file in turn_files %}
{% include turn_file %}
{% endfor %}

<user> {{ query }}