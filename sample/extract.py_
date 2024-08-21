def extract_text_from_table_json(json_data):
    texts = []
    texts.append(json_data['name'])
    texts.append(json_data['description']) if 'description' in json_data else None
    texts.append(json_data['service']['displayName'])

    for column in json_data['columns']:
        texts.append(column['name'])
        # texts.append(column['dataType'])
        texts.append(column['description']) if 'description' in column else None
        for tag in column.get('tags', []):
            texts.append(tag['tagFQN'])
            texts.append(tag['description'])

    for constraint in json_data.get('tableConstraints', []):
        texts.append(constraint['constraintType'])
        texts.extend(constraint['columns'])
        texts.extend(constraint.get('referredColumns', []))

    for tag in json_data.get('tags', []):
        texts.append(tag['tagFQN'])
        texts.append(tag['description'])

    return ' '.join(texts), json_data['fullyQualifiedName']
