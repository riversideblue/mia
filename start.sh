#!/bin/bash

# 変数にAPIキーと都市名を設定
API_KEY="60c24849cbe1845147fd900be9c03593"
HERE="Tokyo"

# APIリクエストを送信して地理情報を取得
GEO_DATA=$(curl -s "http://api.openweathermap.org/geo/1.0/direct?q=$HERE&limit=1&appid=$API_KEY")

# 緯度と経度を抽出
LAT=$(echo $GEO_DATA | jq '.[0].lat')
LON=$(echo $GEO_DATA | jq '.[0].lon')

# APIリクエストを送信して天気情報を取得
WEATHER_DATA=$(curl -s "https://api.openweathermap.org/data/2.5/weather?lat=$LAT&lon=$LON&appid=$API_KEY&units=metric")

# JSONデータから必要な情報を抽出
CITY=$(echo $GEO_DATA | jq -r '.[0].name')
TEMP=$(echo $WEATHER_DATA | jq '.main.temp')
WEATHER=$(echo $WEATHER_DATA | jq -r '.weather[0].main')

# 天気に応じたASCIIアートを選択
case $WEATHER in
    Clear)
        WEATHER_ART=("   \\  |  /    " " .-\"\"\"\"\"-.  " " /        \\ " " \\        / " "  '-.____.-' ")
        ;;
    Clouds)
        WEATHER_ART=("     .--.     " "  .-(    ).  " " (___.__)__) ")
        ;;
    Rain)
        WEATHER_ART=("     .-.      " "    (   ).   " "   (___(__)  " "  ' ' ' ' ' ' " "  ' ' ' ' ' ' ")
        ;;
    Snow)
        WEATHER_ART=("     .-.      " "    (   ).   " "   (___(__)  " "  * * * * *  " "  * * * * *  ")
        ;;
    *)
        WEATHER_ART=("    天気不明 ")
        ;;
esac

# NIDS-CODDのASCIIアート
NIDS_ART=(
    " ____   ____  ___     _____           __  ___    ___   "
    "|    \\ l    j|   \\   / ___/          /  ]|   \\  |   \\  "
    "|  _  Y |  T |    \\ (   \\_  _____   /  / |    \\ |    \\ "
    "|  |  | |  | |  D  Y \\__  T|     | /  /  |  D  Y|  D  Y"
    "|  |  | |  | |     | /  \\ |l_____j/   \\_ |     ||     |"
    "|  |  | j  l |     | \\    |       \\     ||     ||     |"
    "l__j__j|____jl_____j  \\___j        \\____jl_____jl_____j"
)

# 天気情報を表示する配列
WEATHER_INFO=(
    "[${CITY}]"
    " ${WEATHER}"
    " ${TEMP}°C"
)

# 両方のアートと天気情報を並べて表示する
for i in "${!NIDS_ART[@]}"; do
    # NIDSのアートを表示
    printf "%s" "${NIDS_ART[i]}"
    
    # 天気のアートを同じ行に表示（右に3スペース）
    if [ $i -ge 2 ]; then
        printf "   %s" "${WEATHER_ART[i-2]}"
        
        # 天気のアートの隣に天気情報を表示
        if [ $((i - 2)) -lt ${#WEATHER_INFO[@]} ]; then
            printf "   %s" "${WEATHER_INFO[i-2]}"
        fi
    fi
    echo ""
done

echo ""
echo "Host: $(hostname)"
echo "Kernel: $(uname -r)"
echo ""

# シェルを起動
exec /bin/bash